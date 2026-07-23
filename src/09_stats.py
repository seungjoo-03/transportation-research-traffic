# -*- coding: utf-8 -*-
"""가설 정식 판정 (분석 3단계): H2·H3·H4를 사전지정 기준(상세MD §5)으로 검정.

07·08의 산출물(작은 CSV) 위에서 도는 가벼운 계산.
  H2  교차로 간 임계값이 통계 잡음 이상으로 다른가
      기준: 교차로 간 SD의 부트스트랩 95% CI 하한 > 교차로 내(반분할) SD
      교차로 내 SD 근사: 반분할 임계값 차 Δ=qA−qB에서 SD(전체표본 q) ≈ sd(Δ)/2
      (Var(Δ)=2·Var(q_half), 표본 2배면 분산 절반 → sd(q_full)=sd(Δ)/2)
  H3  LOIO 판정 불일치가 잡음 바닥 분포를 벗어나는가
      기준: 교차로별 LOIO 이동량 > 잡음바닥 이동량 p95 / Jaccard < 잡음바닥 p05
  H4  현지 보정 오차 < 무보정(통합) 오차
      기준: 짝지은 단측 Wilcoxon(α=0.05), 중앙값 차 병기

출력: data/processed/stats/hypothesis_tests.csv (+콘솔 판정문)
사용: python src/09_stats.py
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

SEED = 20260723
B_BOOT = 10000


def h2_test(thr: pd.DataFrame, nf_raw: pd.DataFrame, col: str, indicator: str) -> dict:
    q = thr.set_index("intersection")[col].dropna()
    between_sd = q.std(ddof=1)
    rng = np.random.default_rng(SEED)
    boots = [q.sample(len(q), replace=True, random_state=rng.integers(1e9)).std(ddof=1)
             for _ in range(B_BOOT)]
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    d = nf_raw[nf_raw["indicator"] == indicator]
    # 교차로별 반분할 Δ의 SD → 전체표본 임계값의 표본 SD 근사(sd(Δ)/2), 교차로 중앙값
    within_sd = d.groupby("intersection")["shift"].std().median() / 2.0
    return {"indicator": indicator, "hypothesis": "H2",
            "between_sd": round(between_sd, 4), "ci_lo": round(ci_lo, 4), "ci_hi": round(ci_hi, 4),
            "within_sd": round(within_sd, 4),
            "supported": bool(ci_lo > within_sd),
            "detail": f"CI하한 {ci_lo:.3f} vs 교차로내 SD {within_sd:.3f}"}


def h3_test(lo: pd.DataFrame, nf: pd.DataFrame, indicator: str) -> dict:
    d = lo[lo["indicator"] == indicator].merge(
        nf[nf["indicator"] == indicator], on="intersection", how="inner")
    d = d.dropna(subset=["shift_pooled", "shift_p95"])
    n = len(d)
    exceed = int((d["shift_pooled"] > d["shift_p95"]).sum())
    jac_below = int((d["jac_pooled"] < d["jaccard_p05"]).sum())
    return {"indicator": indicator, "hypothesis": "H3",
            "n": n, "shift_exceed_p95": exceed, "jaccard_below_p05": jac_below,
            "supported": bool(exceed > n / 2),
            "detail": f"이동량 잡음초과 {exceed}/{n}곳, 일치도 잡음미달 {jac_below}/{n}곳"}


def h4_test(lo: pd.DataFrame, indicator: str) -> dict:
    d = lo[lo["indicator"] == indicator].dropna(subset=["shift_pooled", "shift_local"])
    stat = wilcoxon(d["shift_local"], d["shift_pooled"], alternative="less")
    med_l, med_p = d["shift_local"].median(), d["shift_pooled"].median()
    better = int((d["shift_local"] < d["shift_pooled"]).sum())
    return {"indicator": indicator, "hypothesis": "H4",
            "n": len(d), "p_value": round(stat.pvalue, 5),
            "median_local": round(med_l, 3), "median_pooled": round(med_p, 3),
            "supported": bool(stat.pvalue < 0.05),
            "detail": f"보정<무보정 {better}/{len(d)}곳, 중앙 {med_l:.3f} vs {med_p:.3f}s, p={stat.pvalue:.4f}"}


def main():
    out = "data/processed/stats"
    os.makedirs(out, exist_ok=True)
    thr = pd.read_csv("data/processed/thresholds/thresholds.csv")
    nf = pd.read_csv("data/processed/thresholds/noise_floor.csv")
    nf_raw = pd.read_csv("data/processed/thresholds/noise_floor_raw.csv")
    lo = pd.read_csv("data/processed/transfer/loio.csv")

    rows = []
    for col, ind in (("ttc_q5", "TTC"), ("pet_q5", "PET")):
        rows.append(h2_test(thr, nf_raw, col, ind))
    for ind in ("TTC", "PET"):
        rows.append(h3_test(lo, nf, ind))
        rows.append(h4_test(lo, ind))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out, "hypothesis_tests.csv"), index=False, encoding="utf-8-sig")
    print("=== 가설 정식 판정 (사전지정 기준, 상세MD §5) ===")
    for _, r in res.iterrows():
        mark = "지지" if r["supported"] else "기각/불충분"
        print(f"[{r['hypothesis']}·{r['indicator']}] {mark} — {r['detail']}")
    print(f"\n저장: {out}/hypothesis_tests.csv")


if __name__ == "__main__":
    main()
