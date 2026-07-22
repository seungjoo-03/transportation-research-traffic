# -*- coding: utf-8 -*-
"""TTC·PET 상충 사건표 (전처리 3단계 — 핵심).

TRIP(Espadaler-Clapés et al. 2025) 3단계를 따른다: ①근접쌍 탐지(속도비례 부채꼴)
→ ②유형 분류(추종/교차) → ③SSM 산출(추종→TTC 연속, 교차→PET 단일값).
구현 세부는 references/methodology/06_구현규칙_확정안.md 에서 확정
(Saunier 코드·Beauchamp 석사논문·SSAM 보고서 원문 대조).

규칙 요약 (출처: T=TRIP 명시, S=원출처, P=사전지정+민감도)
  [공통] 상호작용 계산 전 5Hz 다운샘플(T). 속도벡터는 5Hz 위치차분 후
         2초(10샘플) 이동평균(T). 세그먼트(2초 초과 공백)는 갈라서 처리(P).
  [근접쌍] 부채꼴 r=5v/3.6[m], α=min(100+0.8v,180)°는 전체각(진행방향 ±α/2)(T).
  [추종→TTC] θv<30°∧θp<30°(T; 앞차 정지시 θv 면제=P, TRIP Fig.11 정지 선두차 사례 근거)
         + 접근(반경방향 상대속도 ḋ<-0.1m/s, 원시차분 금지=P, 오차전파 근거)
         + v_sub>v_surr. TTC=중심거리/(v_sub−v_surr)(T). 연속 3샘플(0.4s) 이상만
         이벤트(P). 이벤트별 최소 TTC≤10s만 저장(P; 문헌 최대 임계 4s의 2.5배).
  [교차→PET] 두 궤적 폴리라인의 선분 교차점(P; TI getIntersections 패턴 준용)
         + 통과시각 선형보간(P; 0.2s 양자화 오차 제거). 교차점 방위각차를 저장해
         분석에서 ≥30° 필터(T). 중심점 PET와 차량길이 보정 PET(SSAM 의미론:
         선행차 뒤끝 이탈→후행차 앞끝 도착) 병기(S). 복수 교차점은 최소 채택(S).
         폴리라인 비교차 쌍은 거리≤1.7m 근접 폴백(S; Beauchamp) 후 min|Δt|.
         |Δt|≤10s만(P). 방위각차<15°(동일차로 미세교차=추종 상황)는 저장 생략(P;
         분석 필터 30°에 여유 버퍼).
  [원값 우선] 임계값 컷은 여기서 안 한다 — 원값 저장, 임계값은 분석 단계(p=5%).

출력 (data/processed/conflicts/):
  ttc_events.csv   추종 상충 이벤트(세션·쌍·최소TTC·위치·이동류)
  pet_events.csv   교차 상충 이벤트(세션·쌍·PET중심/보정·방위각차·위치·이동류)
  conflicts_summary.csv  세션별 집계
사용: python src/06_conflicts.py [입력=data/interim] [출력=data/processed]
      python src/06_conflicts.py plots   (그림만 재생성)
"""
import glob
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ---- 파라미터 (사전지정 — 민감도 목록은 확정안 §4) ----
BIN_S = 0.2              # 5Hz
SEG_GAP_S = 2.0          # 세그먼트 경계(03과 동일)
SPEED_MIN = 0.5          # m/s — 미만이면 방향 미정의(정지 취급)
COS30 = np.cos(np.radians(30.0))
EPS_CLOSING = 0.1        # m/s — 접근 판정 데드밴드
MIN_RUN = 3              # TTC 이벤트 최소 연속 샘플(0.4s)
TTC_SAMPLE_CAP = 30.0    # 샘플 보존 상한(이벤트화용, 어떤 임계보다 훨씬 큼)
TTC_EVENT_CAP = 10.0     # 이벤트 최소TTC 저장 상한
PET_CAP = 10.0           # PET 저장 상한
PROX_D = 1.7             # m — 비교차 근접 폴백(Beauchamp)
AZ_STORE_MIN = 15.0      # deg — 이 미만 교차는 저장 생략(분석 필터 30°의 버퍼)
CELL = 3.0               # m — PET 후보쌍 공간격자
LEN_DEFAULT = {0: 4.5, 1: 11.0, 2: 8.0, 3: 2.2}   # 차종별 길이 기본값(결측 보완)

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("mv05", os.path.join(_here, "05_movement.py"))
mv05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mv05)


def local_time_to_sec(series: pd.Series) -> pd.Series:
    parts = series.str.split(":", expand=True)
    return (parts[0].astype("int32") * 3600 + parts[1].astype("int32") * 60
            + parts[2].astype("float64"))


def to_5hz(df: pd.DataFrame) -> pd.DataFrame:
    """30fps → 5Hz 다운샘플 + 세그먼트 분할 + 평활 속도벡터.
    같은 0.2초 빈에서 빈 중심에 가장 가까운 프레임 채택(차량 간 시각 불일치 최소화)."""
    d = df.copy()
    d["t"] = local_time_to_sec(d["Local_Time"])
    d["kbin"] = np.round(d["t"] / BIN_S).astype(np.int64)
    d["err"] = (d["t"] - d["kbin"] * BIN_S).abs()
    d = (d.sort_values("err").drop_duplicates(["Vehicle_ID", "kbin"])
         .sort_values(["Vehicle_ID", "kbin"]).reset_index(drop=True))
    g = d.groupby("Vehicle_ID", sort=False)
    dt = g["t"].diff()
    d["seg"] = (dt > SEG_GAP_S).groupby(d["Vehicle_ID"]).cumsum()
    g2 = d.groupby(["Vehicle_ID", "seg"], sort=False)
    dt2 = g2["t"].diff()
    ok = dt2.between(0.1, 0.4)
    d["vx"] = np.where(ok, g2["Local_X"].diff() / dt2, np.nan)
    d["vy"] = np.where(ok, g2["Local_Y"].diff() / dt2, np.nan)
    for c in ("vx", "vy"):    # 2초(10샘플) 이동평균 — TRIP 방위각 안정화 준용
        d[c] = (d.groupby(["Vehicle_ID", "seg"], sort=False)[c]
                .transform(lambda s: s.rolling(10, center=True, min_periods=2).mean()))
    d["spd"] = np.hypot(d["vx"], d["vy"])
    return d[["Vehicle_ID", "seg", "kbin", "t", "Local_X", "Local_Y", "vx", "vy", "spd"]]


# ---------------- TTC (추종) ----------------

def ttc_samples(d5: pd.DataFrame) -> pd.DataFrame:
    """빈(0.2s)마다: 부채꼴 후보 → 추종 게이트 → 접근 → TTC. 표본행 반환."""
    rows = []
    for kbin, gb in d5.groupby("kbin", sort=True):
        gb = gb.dropna(subset=["vx", "vy"])
        n = len(gb)
        if n < 2:
            continue
        pos = gb[["Local_X", "Local_Y"]].to_numpy()
        vel = gb[["vx", "vy"]].to_numpy()
        spd = gb["spd"].to_numpy()
        vid = gb["Vehicle_ID"].to_numpy()
        subs = np.where(spd >= SPEED_MIN)[0]        # 정지 차량은 sub가 못 됨(r=0)
        if len(subs) == 0:
            continue
        dp = pos[None, :, :] - pos[subs, None, :]   # (m,n,2) sub→surr
        dist = np.hypot(dp[..., 0], dp[..., 1])
        np.putmask(dist, dist == 0, np.inf)         # 자기 자신 제외
        r = 5.0 * spd[subs]                          # r=5v[m/s·s]=5v[km/h]/3.6
        alpha = np.minimum(100.0 + 0.8 * spd[subs] * 3.6, 180.0)
        cos_half = np.cos(np.radians(alpha / 2.0))
        vs = vel[subs]                               # (m,2)
        ns_ = np.hypot(vs[:, 0], vs[:, 1])
        cos_fan = (vs[:, None, 0] * dp[..., 0] + vs[:, None, 1] * dp[..., 1]) / (ns_[:, None] * dist)
        in_fan = (dist <= r[:, None]) & (cos_fan >= cos_half[:, None])
        if not in_fan.any():
            continue
        # θv (앞차 정지시 면제) / θp
        nrm = np.hypot(vel[:, 0], vel[:, 1])
        cos_v = (vs[:, None, 0] * vel[None, :, 0] + vs[:, None, 1] * vel[None, :, 1]) / (ns_[:, None] * np.where(nrm == 0, np.inf, nrm)[None, :])
        ok_v = (cos_v >= COS30) | (spd[None, :] < SPEED_MIN)
        vsum = vs[:, None, :] + vel[None, :, :]
        nsum = np.hypot(vsum[..., 0], vsum[..., 1])
        cos_p = (dp[..., 0] * vsum[..., 0] + dp[..., 1] * vsum[..., 1]) / (dist * np.where(nsum == 0, np.inf, nsum))
        ok_p = cos_p >= COS30
        # 접근(ḋ<-ε) + 속력차
        dv = vel[None, :, :] - vs[:, None, :]        # surr − sub
        ddot = (dp[..., 0] * dv[..., 0] + dp[..., 1] * dv[..., 1]) / dist
        rel = spd[subs][:, None] - spd[None, :]      # v_sub − v_surr
        gate = in_fan & ok_v & ok_p & (ddot < -EPS_CLOSING) & (rel > 0)
        if not gate.any():
            continue
        ttc = dist / rel
        mi, ni = np.where(gate & (ttc <= TTC_SAMPLE_CAP))
        for a, b in zip(mi, ni):
            si = subs[a]
            rows.append((kbin, vid[si], vid[b], ttc[a, b], dist[a, b], ddot[a, b],
                         pos[si, 0], pos[si, 1], spd[si], spd[b]))
    return pd.DataFrame(rows, columns=["kbin", "sub", "surr", "ttc", "dist", "ddot",
                                       "x", "y", "spd_sub", "spd_surr"])


def ttc_eventize(s: pd.DataFrame) -> pd.DataFrame:
    """(sub,surr) 쌍별 연속 구간(빈 간격 1) ≥MIN_RUN → 이벤트. 최소TTC≤캡만."""
    if s.empty:
        return pd.DataFrame()
    ev = []
    for (a, b), g in s.groupby(["sub", "surr"], sort=False):
        g = g.sort_values("kbin")
        run = (g["kbin"].diff() != 1).cumsum()
        for _, r in g.groupby(run):
            if len(r) < MIN_RUN:
                continue
            i = r["ttc"].idxmin()
            if r.loc[i, "ttc"] > TTC_EVENT_CAP:
                continue
            ev.append({"sub": a, "surr": b, "n_samples": len(r),
                       "dur_s": round(len(r) * BIN_S, 1),
                       "min_ttc": round(r.loc[i, "ttc"], 3),
                       "t_min": round(r.loc[i, "kbin"] * BIN_S, 1),
                       "dist_min": round(r.loc[i, "dist"], 2),
                       "x": round(r.loc[i, "x"], 2), "y": round(r.loc[i, "y"], 2),
                       "spd_sub": round(r.loc[i, "spd_sub"] * 3.6, 1),
                       "spd_surr": round(r.loc[i, "spd_surr"] * 3.6, 1)})
    return pd.DataFrame(ev)


# ---------------- PET (교차) ----------------

def _crossings_vec(Pa, Pb, box=30.0):
    """두 폴리라인의 모든 선분교차를 벡터화로 찾는다 → (i, j, rp, rq) 배열.
    box: 세그먼트 중점 간 이 거리(m) 초과 조합은 계산 생략(성긴 프리필터)."""
    A1, A2 = Pa[:-1], Pa[1:]
    B1, B2 = Pb[:-1], Pb[1:]
    ma = (A1 + A2) / 2
    mb = (B1 + B2) / 2
    near = (np.abs(ma[:, None, 0] - mb[None, :, 0]) < box) & \
           (np.abs(ma[:, None, 1] - mb[None, :, 1]) < box)
    ii, jj = np.where(near)
    if len(ii) == 0:
        return np.empty((0, 4))
    d1 = A2[ii] - A1[ii]
    d2 = B2[jj] - B1[jj]
    den = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    ok = np.abs(den) > 1e-12
    w = B1[jj] - A1[ii]
    rp = np.where(ok, (w[:, 0] * d2[:, 1] - w[:, 1] * d2[:, 0]) / np.where(ok, den, 1), -1)
    rq = np.where(ok, (w[:, 0] * d1[:, 1] - w[:, 1] * d1[:, 0]) / np.where(ok, den, 1), -1)
    hit = ok & (rp >= 0) & (rp <= 1) & (rq >= 0) & (rq <= 1)
    return np.column_stack([ii[hit], jj[hit], rp[hit], rq[hit]])


def _cross_time(tarr, i, r):
    return tarr[i] + r * (tarr[i + 1] - tarr[i])


def _time_at_arc(cum, tarr, s_target):
    """호장 s_target 도달 시각(선형보간). 범위 밖이면 None."""
    if s_target <= cum[0]:
        return tarr[0]
    if s_target > cum[-1]:
        return None
    i = int(np.searchsorted(cum, s_target))
    if cum[i] == cum[i - 1]:
        return tarr[i - 1]
    f = (s_target - cum[i - 1]) / (cum[i] - cum[i - 1])
    return tarr[i - 1] + f * (tarr[i] - tarr[i - 1])


def pet_events(d5: pd.DataFrame, veh_len: dict) -> pd.DataFrame:
    """궤적(세그먼트별 폴리라인) 쌍의 교차 PET + 비교차 근접 폴백."""
    trajs = {}
    for (v, sg), g in d5.groupby(["Vehicle_ID", "seg"], sort=False):
        if len(g) < 3:
            continue
        P = g[["Local_X", "Local_Y"]].to_numpy()
        t = g["t"].to_numpy()
        V = g[["vx", "vy"]].to_numpy()
        cum = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1])))))
        trajs[(v, sg)] = (P, t, V, cum)
    # 후보쌍: 공간격자 셀 공유
    cellmap = {}
    for key, (P, *_rest) in trajs.items():
        cells = set(zip((P[:, 0] // CELL).astype(int), (P[:, 1] // CELL).astype(int)))
        for c in cells:
            cellmap.setdefault(c, set()).add(key)
    cand = set()
    for keys in cellmap.values():
        keys = sorted(keys)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                if keys[i][0] != keys[j][0]:      # 다른 차량만
                    cand.add((keys[i], keys[j]))
    ev = []
    from scipy.spatial import cKDTree
    for ka, kb in cand:
        Pa, ta, Va, ca = trajs[ka]
        Pb, tb, Vb, cb = trajs[kb]
        # 시간창 가지치기: 관측 시간대가 PET_CAP 이상 벌어진 쌍은 PET 불가능
        if ta[0] - tb[-1] > PET_CAP or tb[0] - ta[-1] > PET_CAP:
            continue
        best = None
        # 1차: 선분 교차 (벡터화)
        for i, j, rp, rq in _crossings_vec(Pa, Pb):
            i, j = int(i), int(j)
            t1 = _cross_time(ta, i, rp)
            t2 = _cross_time(tb, j, rq)
            pet_c = abs(t2 - t1)
            if pet_c > PET_CAP:
                continue
            if best is None or pet_c < best["pet_centroid"]:
                v1 = Va[i if rp < 0.5 else i + 1]
                v2 = Vb[j if rq < 0.5 else j + 1]
                az = _angle_deg(v1, v2)
                pt = Pa[i] + rp * (Pa[i + 1] - Pa[i])
                s1 = ca[i] + rp * (ca[i + 1] - ca[i])
                s2 = cb[j] + rq * (cb[j + 1] - cb[j])
                best = {"pet_centroid": pet_c, "x": pt[0], "y": pt[1],
                        "t1": t1, "t2": t2, "azdiff": az, "s1": s1, "s2": s2}
        if best is not None:
            if best["azdiff"] is None or best["azdiff"] < AZ_STORE_MIN:
                continue                                # 동일차로 미세교차 → 저장 생략
            # 차량길이 보정(SSAM: 선행차 뒤끝 이탈 → 후행차 앞끝 도착)
            La = veh_len.get(ka[0], 4.5)
            Lb = veh_len.get(kb[0], 4.5)
            if best["t1"] <= best["t2"]:
                lead, foll = (ta, ca, La, best["s1"]), (tb, cb, Lb, best["s2"])
            else:
                lead, foll = (tb, cb, Lb, best["s2"]), (ta, ca, La, best["s1"])
            t_clear = _time_at_arc(lead[1], lead[0], lead[3] + lead[2] / 2)
            t_arr = _time_at_arc(foll[1], foll[0], foll[3] - foll[2] / 2)
            pet_corr = (t_arr - t_clear) if (t_clear is not None and t_arr is not None) else np.nan
            ev.append({"v1": ka[0], "v2": kb[0], "channel": "crossing",
                       "pet_centroid": round(best["pet_centroid"], 3),
                       "pet_corrected": round(pet_corr, 3) if pd.notna(pet_corr) else np.nan,
                       "azdiff": round(best["azdiff"], 1),
                       "x": round(best["x"], 2), "y": round(best["y"], 2),
                       "t1": round(best["t1"], 2), "t2": round(best["t2"], 2)})
        else:
            # 2차 폴백: 거리≤1.7m 최소시간차 (Beauchamp)
            tree = cKDTree(Pb)
            dd, jj = tree.query(Pa, distance_upper_bound=PROX_D)
            hits = np.where(np.isfinite(dd))[0]
            if len(hits) == 0:
                continue
            dts = np.abs(ta[hits] - tb[jj[hits]])
            k = int(np.argmin(dts))
            if dts[k] > PET_CAP:
                continue
            i, j = hits[k], jj[hits[k]]
            az = _angle_deg(Va[i], Vb[j])
            if az is None or az < AZ_STORE_MIN:
                continue
            ev.append({"v1": ka[0], "v2": kb[0], "channel": "proximity",
                       "pet_centroid": round(dts[k], 3), "pet_corrected": np.nan,
                       "azdiff": round(az, 1),
                       "x": round(Pa[i, 0], 2), "y": round(Pa[i, 1], 2),
                       "t1": round(ta[i], 2), "t2": round(tb[j], 2)})
    out = pd.DataFrame(ev)
    if out.empty:
        return out
    # 같은 차량쌍이 세그먼트 여러 개로 잡히면 최소 PET 하나만 (SSAM min 채택)
    out["pair"] = out.apply(lambda r: tuple(sorted((r["v1"], r["v2"]))), axis=1)
    out = out.sort_values("pet_centroid").drop_duplicates("pair").drop(columns="pair")
    return out.reset_index(drop=True)


def _angle_deg(v1, v2):
    n1, n2 = np.hypot(*v1), np.hypot(*v2)
    if not np.isfinite(n1) or not np.isfinite(n2) or n1 < 1e-6 or n2 < 1e-6:
        return None
    c = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
    return float(np.degrees(np.arccos(c)))


# ---------------- 세션 처리 ----------------

USECOLS = ["Vehicle_ID", "Local_Time", "Local_X", "Local_Y",
           "Vehicle_Length", "Vehicle_Class", "Road_Section"]


def process_session(csv_path: str, ang: dict) -> tuple:
    df = pd.read_csv(csv_path, usecols=USECOLS, low_memory=False)
    # 차량 길이: 실측(중앙값), 결측은 차종 기본값
    ln = df.groupby("Vehicle_ID").agg(L=("Vehicle_Length", "median"),
                                      cls=("Vehicle_Class", "first"))
    veh_len = {v: (r["L"] if pd.notna(r["L"]) else LEN_DEFAULT.get(r["cls"], 4.5))
               for v, r in ln.iterrows()}
    mt = mv05.movement_table(df, ang)                       # 이동류 라벨(05 재사용)
    d5 = to_5hz(df)
    ts = ttc_samples(d5)
    tev = ttc_eventize(ts)
    pev = pet_events(d5, veh_len)
    stem = os.path.basename(csv_path)[:-4]
    date, inter, sess = stem.rsplit("_", 2)
    for e, cols in ((tev, [("sub", "sub"), ("surr", "surr")]),
                    (pev, [("v1", "v1"), ("v2", "v2")])):
        if e.empty:
            continue
        e.insert(0, "session", sess)
        e.insert(0, "intersection", inter)
        e.insert(0, "date", date)
        for name, col in cols:
            e[f"mov_{name}"] = e[col].map(mt["movement"])
    return tev, pev


def _work(f: str) -> tuple:
    """병렬 워커: 세션 하나 처리."""
    inter = os.path.basename(f)[:-4].rsplit("_", 2)[1]
    ang = mv05.load_approaches(inter)[0]
    tev, pev = process_session(f, ang)
    return os.path.basename(f)[:-4], tev, pev


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "data/interim"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/processed"
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else max(1, min(8, (os.cpu_count() or 4) - 2))
    outdir = os.path.join(out, "conflicts")
    os.makedirs(outdir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src, "*", "*.csv")))
    print(f"세션 {len(files)}개 상충 계산 시작 (워커 {workers})", flush=True)
    from concurrent.futures import ProcessPoolExecutor
    T, P, summ = [], [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (stem, tev, pev) in enumerate(ex.map(_work, files, chunksize=2), 1):
            T.append(tev)
            P.append(pev)
            summ.append({"file": stem, "ttc_events": len(tev), "pet_events": len(pev)})
            if i % 50 == 0 or i == len(files):
                print(f"  {i}/{len(files)}", flush=True)
    tall = pd.concat([x for x in T if not x.empty], ignore_index=True)
    pall = pd.concat([x for x in P if not x.empty], ignore_index=True)
    tall.to_csv(os.path.join(outdir, "ttc_events.csv"), index=False, encoding="utf-8-sig")
    pall.to_csv(os.path.join(outdir, "pet_events.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(summ).to_csv(os.path.join(outdir, "conflicts_summary.csv"),
                              index=False, encoding="utf-8-sig")
    print(f"\n=== 전체 ===")
    print(f"TTC 이벤트 {len(tall):,} / PET 이벤트 {len(pall):,}")
    print(f"저장: {outdir}/")
    plot_summary(tall, pall)
    plot_validation()
    print("그림 저장: outputs/conflicts/")


def plot_validation(csv_path: str = "data/interim/2022-10-04_A/2022-10-04_A_AM1.csv",
                    outdir: str = "outputs/conflicts") -> None:
    """샘플 세션에서 ① 최심 TTC 이벤트의 거리·접근속도·TTC 3단 시계열
    (TRIP Fig.11의 '거리 단조감소 + TTC 볼록' 거동 재현 확인) ② PET 교차 사례."""
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(csv_path, usecols=USECOLS, low_memory=False)
    d5 = to_5hz(df)
    ts = ttc_samples(d5)
    tev = ttc_eventize(ts)
    if not tev.empty:
        e = tev.loc[tev["min_ttc"].idxmin()]
        s = ts[(ts["sub"] == e["sub"]) & (ts["surr"] == e["surr"])].sort_values("kbin")
        s = s[(s["kbin"] * BIN_S).between(e["t_min"] - 15, e["t_min"] + 10)]
        tt = s["kbin"] * BIN_S - e["t_min"]
        fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
        axes[0].plot(tt, s["dist"], color="#4C78A8")
        axes[0].set_ylabel("거리 (m)")
        axes[1].plot(tt, s["ddot"], color="#F58518")
        axes[1].axhline(-EPS_CLOSING, color="gray", ls=":", lw=1)
        axes[1].set_ylabel("반경방향 상대속도 (m/s)")
        axes[2].plot(tt, s["ttc"], color="crimson")
        axes[2].set_ylabel("TTC (초)")
        axes[2].set_xlabel(f"최소 TTC 시점 기준 상대시간 (초) — 최소 TTC {e['min_ttc']:.2f}s")
        axes[0].set_title(f"검증: 최심 추종 상충의 시계열 (차량 {e['sub']}→{e['surr']})")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "검증_TTC사례.png"), dpi=120)
        plt.close(fig)
    ln = df.groupby("Vehicle_ID").agg(L=("Vehicle_Length", "median"), cls=("Vehicle_Class", "first"))
    veh_len = {v: (r["L"] if pd.notna(r["L"]) else LEN_DEFAULT.get(r["cls"], 4.5))
               for v, r in ln.iterrows()}
    pev = pet_events(d5, veh_len)
    pc = pev[(pev["channel"] == "crossing") & (pev["azdiff"] >= 30)]
    if not pc.empty:
        e = pc.loc[pc["pet_centroid"].idxmin()]
        fig, ax = plt.subplots(figsize=(7, 7))
        for v, col in ((e["v1"], "#4C78A8"), (e["v2"], "#E45756")):
            g = d5[d5["Vehicle_ID"] == v]
            ax.plot(g["Local_X"], g["Local_Y"], color=col, lw=1.5, label=f"차량 {v}")
        ax.plot(e["x"], e["y"], "k*", ms=16, label="교차점")
        ax.set_aspect("equal")
        ax.set_title(f"검증: 최심 교차 상충 — PET {e['pet_centroid']:.2f}s, 방위각차 {e['azdiff']:.0f}°")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "검증_PET사례.png"), dpi=120)
        plt.close(fig)


def plot_summary(tall: pd.DataFrame, pall: pd.DataFrame, outdir="outputs/conflicts") -> None:
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(tall["min_ttc"], bins=40, color="#4C78A8", edgecolor="white")
    ax.set_xlabel("이벤트 최소 TTC (초)")
    ax.set_ylabel("이벤트 수")
    ax.set_title(f"추종 상충 TTC 분포 (이벤트 {len(tall):,}건)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "분포_TTC.png"), dpi=120); plt.close(fig)

    pc = pall[pall["azdiff"] >= 30]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pc["pet_centroid"], bins=40, color="#E45756", edgecolor="white")
    ax.set_xlabel("PET (초, 중심점 기준, 방위각차≥30°)")
    ax.set_ylabel("이벤트 수")
    ax.set_title(f"교차 상충 PET 분포 (이벤트 {len(pc):,}건)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "분포_PET.png"), dpi=120); plt.close(fig)

    cnt = (tall.groupby("intersection").size().rename("TTC").to_frame()
           .join(pc.groupby("intersection").size().rename("PET")).fillna(0))
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(cnt))
    ax.bar(x - 0.2, cnt["TTC"], width=0.4, color="#4C78A8", label="TTC(추종)")
    ax.bar(x + 0.2, cnt["PET"], width=0.4, color="#E45756", label="PET(교차)")
    ax.set_xticks(x); ax.set_xticklabels(cnt.index)
    ax.set_xlabel("교차로"); ax.set_ylabel("이벤트 수(4일 합)")
    ax.set_title("교차로별 상충 이벤트 수")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "분포_교차로별상충.png"), dpi=120); plt.close(fig)


if __name__ == "__main__":
    matplotlib.use("Agg")
    if len(sys.argv) > 1 and sys.argv[1] == "plots":
        base = "data/processed/conflicts"
        tall = pd.read_csv(os.path.join(base, "ttc_events.csv"))
        pall = pd.read_csv(os.path.join(base, "pet_events.csv"))
        plot_summary(tall, pall)
        plot_validation()
        print("그림 저장: outputs/conflicts/")
    else:
        main()