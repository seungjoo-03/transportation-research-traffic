# -*- coding: utf-8 -*-
"""보완 분석 (빠짐 감사 결과 4+1건): 사전 지정 산출물 완결.

① PET 고정 임계값 스윕 — 계획서 명시("통용 대표값 부재 → 스윕 곡선 전체 제시").
   참조 2종: SSAM 기본값 5.0s 기준 τ + 인접 스텝(±0.5s) τ 곡선.
② H5 시간대 축 — 교차로별 오전(AM) vs 오후(PM) 임계값(q5) 짝 비교(Wilcoxon).
③ 교통량 보정 H6 — 시간당 대신 통과차량 1천대당 상충률 순위로 사고 상관 재확인
   ("교통량 효과" 반론 방어). 통과차량 = movement_summary의 직진+좌회전+우회전.
④ 1.5초 백분위 — TTC 1.5s가 각 교차로 사건 분포에서 차지하는 백분위(착수 전 확인 3).
⑤ (탐색적) 사고유형 짝 대조 — TTC(추종) 순위↔추돌 사고, PET(교차) 순위↔측면직각 사고.

출력: data/processed/stats/supplement.csv 외, 그림 outputs/transfer/분포_PET순위안정성.png,
      outputs/thresholds/분포_시간대별임계값.png
사용: python src/13_supplement.py
"""
import importlib.util
import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pyproj import Transformer
from scipy.stats import kendalltau, spearmanr, wilcoxon

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("th07", os.path.join(_here, "07_thresholds.py"))
th07 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(th07)

P = th07.P_MAIN
SWEEP_PET = np.round(np.arange(0.5, 6.01, 0.1), 1)
RADIUS = 50.0


def pet_sweep(pet: pd.DataFrame, expo: pd.Series):
    """① PET 스윕: 임계값별 교차로 상충률 순위 안정성 (vs SSAM 5.0s + 인접 ±0.5s)."""
    inters = sorted(pet["intersection"].unique())
    rates = pd.DataFrame(index=inters, columns=SWEEP_PET, dtype=float)
    for th in SWEEP_PET:
        cnt = pet[pet["pet_centroid"] <= th].groupby("intersection").size()
        rates[th] = cnt.reindex(inters).fillna(0) / expo.reindex(inters)
    base = rates[5.0].rank()                      # SSAM 기본 max PET 5s 참조
    rows = []
    for th in SWEEP_PET:
        t_ssam = kendalltau(base, rates[th].rank()).statistic
        th_adj = round(th + 0.5, 1)
        t_adj = (kendalltau(rates[th].rank(), rates[th_adj].rank()).statistic
                 if th_adj in rates.columns else np.nan)
        rows.append({"threshold": th, "tau_vs_5.0s": round(t_ssam, 3),
                     "tau_adjacent+0.5s": round(t_adj, 3) if pd.notna(t_adj) else np.nan})
    df = pd.DataFrame(rows)
    rates.to_csv("data/processed/transfer/sweep_rate_pet.csv", encoding="utf-8-sig")
    df.to_csv("data/processed/transfer/sweep_tau_pet.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(df["threshold"], df["tau_vs_5.0s"], marker="o", ms=3, color="#E45756",
            label="vs SSAM 기본값 5.0s")
    ax.plot(df["threshold"], df["tau_adjacent+0.5s"], marker="s", ms=3, color="#F58518",
            label="인접 스텝(+0.5s)")
    ax.axhline(0.8, color="crimson", ls=":", label="τ=0.8")
    ax.set_xlabel("고정 PET 임계값 (초)")
    ax.set_ylabel("순위 상관 (Kendall τ)")
    ax.set_title("PET: 고정 임계값 선택에 따른 교차로 위험 순위 안정성")
    ax.legend()
    fig.tight_layout(); fig.savefig("outputs/transfer/분포_PET순위안정성.png", dpi=120); plt.close(fig)
    lit = df[(df["threshold"] >= 1.0) & (df["threshold"] <= 5.0)]
    return {"pet_sweep_min_tau_vs5": float(lit["tau_vs_5.0s"].min()),
            "pet_sweep_min_tau_adj": float(lit["tau_adjacent+0.5s"].min())}


def ampm_h5(ttc: pd.DataFrame, pet: pd.DataFrame):
    """② H5 시간대 축: 교차로별 AM vs PM q5 짝 비교."""
    res = {}
    rows = []
    for ind, df, col in (("TTC", ttc, "min_ttc"), ("PET", pet, "pet_centroid")):
        df = df.copy()
        df["half"] = np.where(df["session"].str.startswith("AM"), "AM", "PM")
        q = (df.groupby(["intersection", "half"])[col]
             .apply(lambda s: s.quantile(P) if len(s) >= 20 else np.nan).unstack())
        q = q.dropna()
        diff = (q["AM"] - q["PM"]).abs()
        try:
            pval = wilcoxon(q["AM"], q["PM"]).pvalue
        except ValueError:
            pval = np.nan
        res[f"{ind}_ampm_n"] = len(q)
        res[f"{ind}_ampm_absdiff_med"] = round(float(diff.median()), 3)
        res[f"{ind}_ampm_wilcoxon_p"] = round(float(pval), 4) if pd.notna(pval) else np.nan
        for inter, r in q.iterrows():
            rows.append({"indicator": ind, "intersection": inter,
                         "AM": round(r["AM"], 3), "PM": round(r["PM"], 3)})
    pd.DataFrame(rows).to_csv("data/processed/stats/ampm_thresholds.csv",
                              index=False, encoding="utf-8-sig")
    # 그림: TTC AM vs PM 산점
    d = pd.DataFrame(rows)
    d = d[d["indicator"] == "TTC"]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(d["AM"], d["PM"], s=45, color="#4C78A8")
    lim = [min(d["AM"].min(), d["PM"].min()) - 0.05, max(d["AM"].max(), d["PM"].max()) + 0.05]
    ax.plot(lim, lim, "r--", lw=1, label="AM = PM")
    for _, r in d.iterrows():
        ax.annotate(r["intersection"], (r["AM"], r["PM"]), fontsize=8,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("오전(AM) 임계값 q5 (초)")
    ax.set_ylabel("오후(PM) 임계값 q5 (초)")
    ax.set_title("H5 시간대 축: 교차로별 TTC 임계값 AM vs PM")
    ax.legend()
    fig.tight_layout(); fig.savefig("outputs/thresholds/분포_시간대별임계값.png", dpi=120); plt.close(fig)
    return res


def volume_adjusted_h6(ttc: pd.DataFrame, pet: pd.DataFrame):
    """③ 교통량 보정 H6 + PET 순위 H6: 사고 상관이 교통량 효과가 아님을 확인."""
    mv = pd.read_csv("data/processed/movement_summary.csv")
    vol = mv.groupby("intersection")[["직진", "좌회전", "우회전"]].sum().sum(axis=1)
    cc = pd.read_csv("data/processed/taas/intersection_crashes.csv").set_index("intersection")
    daily = pd.read_csv("data/processed/intersection_daily.csv")
    expo = daily.groupby("intersection")["effective_min"].sum() / 60.0
    res = {}
    n15 = ttc[ttc["min_ttc"] <= 1.5].groupby("intersection").size()
    pet_n = pet.groupby("intersection").size()
    variants = {
        "TTC1.5_per_hour": n15 / expo,
        "TTC1.5_per_1k_veh": n15 / (vol / 1000.0),
        "PET_per_hour": pet_n / expo,
        "PET_per_1k_veh": pet_n / (vol / 1000.0),
    }
    for name, series in variants.items():
        m = cc.join(series.rename("rate")).dropna(subset=["rate"])
        for crash in ("severe_50m", "all_50m"):
            rho, p = spearmanr(m["rate"], m[crash])
            res[f"h6_{name}_vs_{crash}"] = f"rho={rho:.3f}, p={p:.4f}"
    return res


def pct_15(ttc: pd.DataFrame):
    """④ TTC 1.5초가 각 교차로 사건 분포에서 차지하는 백분위."""
    pct = ttc.groupby("intersection")["min_ttc"].apply(lambda s: (s <= 1.5).mean() * 100)
    return {"pct15_min": round(float(pct.min()), 1),
            "pct15_median": round(float(pct.median()), 1),
            "pct15_max": round(float(pct.max()), 1),
            "pct15_pooled": round(float((ttc["min_ttc"] <= 1.5).mean() * 100), 1)}


def type_matched(ttc: pd.DataFrame, pet: pd.DataFrame):
    """⑤ (탐색적) 사고유형 짝 대조: TTC↔추돌, PET↔측면직각."""
    taas = pd.read_csv("data/processed/taas/taas_matched.csv", low_memory=False)
    near = taas[taas["dist_m"] <= RADIUS]
    types = near["acdnt_mdc"].astype(str)
    rear = near[types.str.contains("추돌", na=False)].groupby("nearest").size()
    side = near[types.str.contains("측면|직각", na=False, regex=True)].groupby("nearest").size()
    daily = pd.read_csv("data/processed/intersection_daily.csv")
    expo = daily.groupby("intersection")["effective_min"].sum() / 60.0
    t_rate = (ttc[ttc["min_ttc"] <= 1.5].groupby("intersection").size() / expo)
    p_rate = (pet.groupby("intersection").size() / expo)
    res = {"n_rear_crash": int(rear.sum()), "n_side_crash": int(side.sum())}
    for name, rate, crash in (("TTC×추돌", t_rate, rear), ("PET×측면직각", p_rate, side),
                              ("TTC×측면직각", t_rate, side), ("PET×추돌", p_rate, rear)):
        m = pd.concat([rate.rename("r"), crash.rename("c")], axis=1).fillna({"c": 0}).dropna()
        rho, p = spearmanr(m["r"], m["c"])
        res[f"type_{name}"] = f"rho={rho:.3f}, p={p:.4f} (n={len(m)})"
    return res


def main():
    ttc, pet = th07.load_events("data/processed/conflicts")
    daily = pd.read_csv("data/processed/intersection_daily.csv")
    expo = daily.groupby("intersection")["effective_min"].sum() / 60.0

    out = {}
    out.update(pet_sweep(pet, expo))
    out.update(ampm_h5(ttc, pet))
    out.update(volume_adjusted_h6(ttc, pet))
    out.update(pct_15(ttc))
    out.update(type_matched(ttc, pet))

    s = pd.Series(out)
    os.makedirs("data/processed/stats", exist_ok=True)
    s.to_csv("data/processed/stats/supplement.csv", encoding="utf-8-sig")
    print("=== 보완 분석 결과 ===")
    for k, v in out.items():
        print(f"  {k}: {v}")
    print("\n저장: data/processed/stats/supplement.csv 외 그림 2장")


if __name__ == "__main__":
    matplotlib.use("Agg")
    main()