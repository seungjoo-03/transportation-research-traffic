# -*- coding: utf-8 -*-
"""사건 수 표 · 자료 기반 임계값 · 잡음 바닥 (분석 1단계).

06의 상충 사건표(작은 CSV) 위에서 도는 가벼운 계산.
① 사건 수 표 — 교차로·세션·일 단위 TTC/PET 이벤트 수. p=5% 분위수를 어느
   단위에서 추정할지 사전 규칙으로 판정한다(셀당 한 자릿수면 단위 상향, 상세MD §4).
② 임계값 — 각 교차로 상충 지표 분포의 하위 p 분위수(주분석 p=5%, 스윕 1~25%).
   TTC는 이벤트 최소TTC, PET는 방위각차 30° 이상 교차 이벤트의 pet_centroid.
③ 잡음 바닥 — 같은 교차로의 세션들을 무작위 반분할(B회 반복), 반쪽 임계값을
   다른 반쪽에 적용했을 때의 임계값 이동량·판정 일치도(Jaccard) 분포.
   진짜 차이가 0인 상황의 불일치 = 표본 잡음 기준선 → H2·H3 판정 근거.

출력 (data/processed/thresholds/): event_counts.csv, thresholds.csv,
  noise_floor.csv, noise_floor_raw.csv
그림 (outputs/thresholds/): 분포_임계값_교차로별.png, 분포_잡음바닥.png,
  분포_사건수.png
사용: python src/07_thresholds.py [상충=data/processed/conflicts] [출력=data/processed/thresholds]
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

P_MAIN = 0.05                    # 주분석 분위수
P_SWEEP = np.arange(0.01, 0.26, 0.01)
AZ_MIN = 30.0                    # PET 유효 방위각차(TRIP)
B_SPLIT = 200                    # 잡음 바닥 반분할 반복 수 (사전지정)
SEED = 20260723                  # 재현성
# 인공물 필터(사전지정·민감도 병기): 중복 검출(한 차가 두 ID) 제거.
#  - TTC: 최소거리 1m 미만 = 차량 중심이 물리적으로 겹침(차폭 1.8m) → 인공물
#  - PET: 0.2s(1샘플) 미만 = 동시각 동일지점 통과 → 실충돌이 아니면 인공물
#  근거: 진단에서 PET<0.1s의 다수가 t1==t2 완전 동시각 + 1.7m 이내 동시 존재,
#  README 'Vehicle ID 모호성' 명시. 민감도: TTC {0.5,1.0,1.5}m, PET {0.1,0.2,0.5}s
TTC_ARTIFACT_DIST_M = 1.0
PET_ARTIFACT_MIN_S = 0.2


def load_events(cdir: str):
    ttc = pd.read_csv(os.path.join(cdir, "ttc_events.csv"))
    n0 = len(ttc)
    ttc = ttc[ttc["dist_min"] >= TTC_ARTIFACT_DIST_M].copy()
    pet = pd.read_csv(os.path.join(cdir, "pet_events.csv"))
    m0 = len(pet)
    pet = pet[(pet["azdiff"] >= AZ_MIN)
              & (pet["pet_centroid"] >= PET_ARTIFACT_MIN_S)].copy()
    print(f"인공물 필터: TTC {n0 - len(ttc):,}건 제거, "
          f"PET(방위각 포함) {m0 - len(pet):,}건 제외", flush=True)
    for d in (ttc, pet):
        d["sess_key"] = d["date"] + "_" + d["session"]
    return ttc, pet


def event_counts(ttc: pd.DataFrame, pet: pd.DataFrame) -> pd.DataFrame:
    """교차로별: 총 이벤트, 세션당·일당 중앙값 → 분위수 추정 단위 판정 재료."""
    rows = []
    for inter in sorted(ttc["intersection"].unique()):
        t = ttc[ttc["intersection"] == inter]
        p = pet[pet["intersection"] == inter]
        rows.append({
            "intersection": inter,
            "ttc_total": len(t),
            "ttc_per_session_med": t.groupby("sess_key").size().median() if len(t) else 0,
            "ttc_per_day_med": t.groupby("date").size().median() if len(t) else 0,
            "pet_total": len(p),
            "pet_per_session_med": p.groupby("sess_key").size().median() if len(p) else 0,
            "pet_per_day_med": p.groupby("date").size().median() if len(p) else 0,
        })
    return pd.DataFrame(rows)


def thresholds_table(ttc: pd.DataFrame, pet: pd.DataFrame) -> pd.DataFrame:
    """교차로별 하위 p 분위수 임계값 (주분석 p=5% + 스윕)."""
    rows = []
    for inter in sorted(ttc["intersection"].unique()):
        r = {"intersection": inter}
        tv = ttc.loc[ttc["intersection"] == inter, "min_ttc"]
        pv = pet.loc[pet["intersection"] == inter, "pet_centroid"]
        r["ttc_n"] = len(tv)
        r["pet_n"] = len(pv)
        r["ttc_q5"] = tv.quantile(P_MAIN) if len(tv) else np.nan
        r["pet_q5"] = pv.quantile(P_MAIN) if len(pv) else np.nan
        for p in P_SWEEP:
            r[f"ttc_q{int(round(p*100))}"] = tv.quantile(p) if len(tv) else np.nan
            r[f"pet_q{int(round(p*100))}"] = pv.quantile(p) if len(pv) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def _split_half_once(df: pd.DataFrame, col: str, rng) -> dict:
    """세션 단위 무작위 반분할 → 임계값 이동량 + 반쪽B 사건 판정 Jaccard."""
    sess = df["sess_key"].unique()
    if len(sess) < 4:
        return None
    rng.shuffle(sess)
    half = len(sess) // 2
    A = df[df["sess_key"].isin(sess[:half])][col]
    B = df[df["sess_key"].isin(sess[half:])][col]
    if len(A) < 20 or len(B) < 20:                 # 분위수 최소 표본(사전지정)
        return None
    qA, qB = A.quantile(P_MAIN), B.quantile(P_MAIN)
    conflict_by_A = B <= qA                        # 반쪽B 사건을 A임계값으로 판정
    conflict_by_B = B <= qB                        # 자기(B) 임계값으로 판정
    inter_ = (conflict_by_A & conflict_by_B).sum()
    union = (conflict_by_A | conflict_by_B).sum()
    return {"shift": abs(qA - qB), "jaccard": inter_ / union if union else 1.0}


def noise_floor(ttc: pd.DataFrame, pet: pd.DataFrame):
    """교차로별 반분할 B회 → 이동량·Jaccard 분포. 원자료와 요약 둘 다 반환."""
    rng = np.random.default_rng(SEED)
    raw = []
    for name, df, col in (("TTC", ttc, "min_ttc"), ("PET", pet, "pet_centroid")):
        for inter in sorted(df["intersection"].unique()):
            sub = df[df["intersection"] == inter]
            for b in range(B_SPLIT):
                r = _split_half_once(sub, col, rng)
                if r:
                    raw.append({"indicator": name, "intersection": inter, "rep": b, **r})
    rawdf = pd.DataFrame(raw)
    summ = (rawdf.groupby(["indicator", "intersection"])
            .agg(n_reps=("shift", "size"),
                 shift_med=("shift", "median"),
                 shift_p95=("shift", lambda s: s.quantile(0.95)),
                 jaccard_med=("jaccard", "median"),
                 jaccard_p05=("jaccard", lambda s: s.quantile(0.05)))
            .reset_index())
    return rawdf, summ


def plots(cnt, thr, nf_raw, outdir="outputs/thresholds"):
    os.makedirs(outdir, exist_ok=True)
    # 사건 수
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(cnt))
    ax.bar(x - 0.2, cnt["ttc_total"], width=0.4, color="#4C78A8", label="TTC")
    ax.bar(x + 0.2, cnt["pet_total"], width=0.4, color="#E45756", label="PET(교차)")
    ax.set_xticks(x); ax.set_xticklabels(cnt["intersection"])
    ax.set_yscale("log")
    ax.set_ylabel("이벤트 수 (4일 합, 로그)")
    ax.set_title("교차로별 상충 이벤트 수")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "분포_사건수.png"), dpi=120); plt.close(fig)
    # 임계값
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].bar(thr["intersection"], thr["ttc_q5"], color="#4C78A8")
    axes[0].set_title("교차로별 TTC 임계값 (하위 5% 분위수)")
    axes[0].set_ylabel("초")
    axes[1].bar(thr["intersection"], thr["pet_q5"], color="#E45756")
    axes[1].set_title("교차로별 PET 임계값 (하위 5% 분위수)")
    axes[1].set_ylabel("초")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "분포_임계값_교차로별.png"), dpi=120); plt.close(fig)
    # 잡음 바닥
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, ind, col in ((axes[0], "TTC", "#4C78A8"), (axes[1], "PET", "#E45756")):
        d = nf_raw[nf_raw["indicator"] == ind]
        if len(d):
            ax.hist(d["shift"], bins=40, color=col, edgecolor="white")
            ax.axvline(d["shift"].quantile(0.95), color="crimson", ls="--",
                       label=f"95백분위 {d['shift'].quantile(0.95):.3f}s")
            ax.legend()
        ax.set_title(f"잡음 바닥: {ind} 임계값 이동량 (같은 교차로 반분할)")
        ax.set_xlabel("|이동량| (초)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "분포_잡음바닥.png"), dpi=120); plt.close(fig)


def main():
    cdir = sys.argv[1] if len(sys.argv) > 1 else "data/processed/conflicts"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/processed/thresholds"
    os.makedirs(out, exist_ok=True)
    ttc, pet = load_events(cdir)
    print(f"TTC {len(ttc):,} / PET(교차) {len(pet):,} 이벤트 로드", flush=True)

    cnt = event_counts(ttc, pet)
    cnt.to_csv(os.path.join(out, "event_counts.csv"), index=False, encoding="utf-8-sig")
    print("\n=== 사건 수 표 (분위수 추정 단위 판정) ===")
    print(cnt.to_string(index=False))
    # 사전 규칙: 셀 중앙값이 한 자릿수면 단위 상향 (세션→일→교차로)
    unit_ttc = "세션" if cnt["ttc_per_session_med"].median() >= 10 else ("교차로-일" if cnt["ttc_per_day_med"].median() >= 10 else "교차로")
    unit_pet = "세션" if cnt["pet_per_session_med"].median() >= 10 else ("교차로-일" if cnt["pet_per_day_med"].median() >= 10 else "교차로")
    print(f"→ 분위수 추정 단위(사전 규칙): TTC={unit_ttc}, PET={unit_pet}")

    thr = thresholds_table(ttc, pet)
    thr.to_csv(os.path.join(out, "thresholds.csv"), index=False, encoding="utf-8-sig")
    print("\n=== 교차로별 임계값 (p=5%) ===")
    print(thr[["intersection", "ttc_n", "ttc_q5", "pet_n", "pet_q5"]].round(3).to_string(index=False))

    nf_raw, nf = noise_floor(ttc, pet)
    nf_raw.to_csv(os.path.join(out, "noise_floor_raw.csv"), index=False, encoding="utf-8-sig")
    nf.to_csv(os.path.join(out, "noise_floor.csv"), index=False, encoding="utf-8-sig")
    print("\n=== 잡음 바닥 (반분할 B=%d) ===" % B_SPLIT)
    print(nf.round(3).to_string(index=False))

    plots(cnt, thr, nf_raw)
    print(f"\n저장: {out}/, 그림: outputs/thresholds/")


if __name__ == "__main__":
    matplotlib.use("Agg")
    main()