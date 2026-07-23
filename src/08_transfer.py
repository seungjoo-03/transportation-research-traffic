# -*- coding: utf-8 -*-
"""임계값 전이 검증 (분석 2단계 — 핵심): LOIO · 쌍별 380 · H1 스윕.

계획서 RQ3·RQ5·H1~H3 구현. 07과 같은 인공물 필터·분위수(p=5%)를 쓴다.

① LOIO 20-fold (주분석, RQ3·H3)
   각 교차로 X에 대해: 나머지 19곳의 10/4~06 사건으로 교차로별 q5를 구해
   그 중앙값을 통합 임계값으로(계획서 §3 풀링) → X의 10/7 사건에 적용.
   비교 기준은 X의 10/7 자기 임계값. 보조로 X의 10/4~06 현지 보정값도 적용.
   지표: 임계값 이동량 |Δq|, 판정 일치도 Jaccard, 상충건수 비.
   판정(H3): LOIO 불일치가 07의 잡음 바닥(같은 교차로 반분할) 분포를 넘는가.
② 쌍별 전이 (보조, RQ5) — i곳 q5를 j곳 사건에 적용, 20×19 이동량·Jaccard 행렬.
   가설·합격선 없이 히트맵으로만 제시(계획서 §4).
③ H1 스윕 — 고정 임계값 0.5~6.0s(0.1 간격)에서 교차로 상충률(노출시간당)
   순위를 구하고, 임계값 쌍 간 Kendall τ로 순위 안정성 측정.
   노출량: intersection_daily.csv의 유효 관측시간 합(4일).

출력 (data/processed/transfer/): loio.csv, pairwise_shift_[ttc|pet].csv,
  pairwise_jaccard_[ttc|pet].csv, sweep_rate_ttc.csv, sweep_tau_ttc.csv
그림 (outputs/transfer/): 분포_LOIO_이동량.png, 분포_LOIO_일치도.png,
  히트맵_쌍별전이.png, 분포_순위안정성.png
사용: python src/08_transfer.py
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("th07", os.path.join(_here, "07_thresholds.py"))
th07 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(th07)

P = th07.P_MAIN                      # 0.05
TRAIN_DATES = ["2022-10-04", "2022-10-05", "2022-10-06"]
TEST_DATE = "2022-10-07"
SWEEP = np.round(np.arange(0.5, 6.01, 0.1), 1)   # H1 고정 임계값 스윕(문헌 범위)


def q5(s: pd.Series) -> float:
    return s.quantile(P) if len(s) >= 20 else np.nan   # 최소 표본 20(07과 동일)


def jaccard(vals: pd.Series, qa: float, qb: float) -> float:
    A = vals <= qa
    B = vals <= qb
    u = (A | B).sum()
    return (A & B).sum() / u if u else np.nan


def loio(df: pd.DataFrame, col: str, indicator: str) -> pd.DataFrame:
    """각 교차로를 남기고 19곳 통합 임계값을 만들어 남긴 곳 10/7에 적용."""
    rows = []
    inters = sorted(df["intersection"].unique())
    train = df[df["date"].isin(TRAIN_DATES)]
    test = df[df["date"] == TEST_DATE]
    q_train = train.groupby("intersection")[col].apply(q5)   # 교차로별 훈련기간 q5
    for X in inters:
        pooled = q_train.drop(index=X).median()              # 19곳 중앙값(풀링)
        local = q_train.get(X, np.nan)                       # X의 10/4~06 현지 보정
        tv = test.loc[test["intersection"] == X, col]
        own = q5(tv)                                         # X의 10/7 자기 기준
        if pd.isna(own) or len(tv) < 20:
            rows.append({"indicator": indicator, "intersection": X, "n_test": len(tv),
                         "q_pooled": pooled, "q_local": local, "q_own": own,
                         "shift_pooled": np.nan, "shift_local": np.nan,
                         "jac_pooled": np.nan, "jac_local": np.nan,
                         "ratio_pooled": np.nan})
            continue
        rows.append({
            "indicator": indicator, "intersection": X, "n_test": len(tv),
            "q_pooled": round(pooled, 3), "q_local": round(local, 3) if pd.notna(local) else np.nan,
            "q_own": round(own, 3),
            "shift_pooled": round(abs(pooled - own), 3),
            "shift_local": round(abs(local - own), 3) if pd.notna(local) else np.nan,
            "jac_pooled": round(jaccard(tv, pooled, own), 3),
            "jac_local": round(jaccard(tv, local, own), 3) if pd.notna(local) else np.nan,
            "ratio_pooled": round((tv <= pooled).sum() / max((tv <= own).sum(), 1), 3),
        })
    return pd.DataFrame(rows)


def pairwise(df: pd.DataFrame, col: str):
    """i곳 q5(전체 4일)를 j곳 사건에 적용 — 이동량·Jaccard 20×20 행렬."""
    inters = sorted(df["intersection"].unique())
    qs = df.groupby("intersection")[col].apply(q5)
    shift = pd.DataFrame(index=inters, columns=inters, dtype=float)
    jac = pd.DataFrame(index=inters, columns=inters, dtype=float)
    for j in inters:
        vals = df.loc[df["intersection"] == j, col]
        for i in inters:
            shift.loc[i, j] = abs(qs[i] - qs[j])
            jac.loc[i, j] = jaccard(vals, qs[i], qs[j]) if i != j else 1.0
    return shift, jac


def sweep_ranks(ttc: pd.DataFrame, expo: pd.Series):
    """H1: 고정 임계값별 교차로 상충률(건/시간) 순위와 순위 안정성(Kendall τ)."""
    inters = sorted(ttc["intersection"].unique())
    rates = pd.DataFrame(index=inters, columns=SWEEP, dtype=float)
    for tau_ in SWEEP:
        cnt = ttc[ttc["min_ttc"] <= tau_].groupby("intersection").size()
        rates[tau_] = cnt.reindex(inters).fillna(0) / expo.reindex(inters)
    taus = []
    base = rates[1.5].rank()                                 # 관행값 1.5s 기준 순위
    for tau_ in SWEEP:
        kt = kendalltau(base, rates[tau_].rank()).statistic
        taus.append({"threshold": tau_, "kendall_tau_vs_1.5s": round(kt, 3)})
    # 인접 스텝 간 안정성도
    for a, b in zip(SWEEP[:-1], SWEEP[1:]):
        pass
    return rates, pd.DataFrame(taus)


def plots(lo, shift_t, jac_t, tau_df, nf, outdir="outputs/transfer"):
    os.makedirs(outdir, exist_ok=True)
    # LOIO 이동량 vs 잡음 바닥
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, ind, col in ((axes[0], "TTC", "#4C78A8"), (axes[1], "PET", "#E45756")):
        d = lo[(lo["indicator"] == ind)].dropna(subset=["shift_pooled"])
        n = nf[nf["indicator"] == ind]
        ax.bar(d["intersection"], d["shift_pooled"], color=col, label="LOIO 이동량")
        if len(n):
            ax.axhline(n["shift_p95"].median(), color="gray", ls="--",
                       label=f"잡음 바닥 p95 (중앙 {n['shift_p95'].median():.3f}s)")
        ax.set_title(f"{ind}: LOIO 임계값 이동량 vs 잡음 바닥")
        ax.set_ylabel("|이동량| (초)")
        ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "분포_LOIO_이동량.png"), dpi=120); plt.close(fig)
    # LOIO 일치도
    fig, ax = plt.subplots(figsize=(11, 4.5))
    w = 0.35
    for k, (ind, col) in enumerate((("TTC", "#4C78A8"), ("PET", "#E45756"))):
        d = lo[lo["indicator"] == ind].dropna(subset=["jac_pooled"])
        x = np.arange(len(d))
        ax.bar(x + (k - 0.5) * w, d["jac_pooled"], width=w, color=col, label=ind)
        ax.set_xticks(x); ax.set_xticklabels(d["intersection"])
    n = nf[nf["indicator"] == "TTC"]
    if len(n):
        ax.axhline(n["jaccard_p05"].median(), color="gray", ls="--",
                   label="잡음 바닥 Jaccard p05(TTC)")
    ax.set_ylim(0, 1.05)
    ax.set_title("LOIO 판정 일치도 (통합 임계값 vs 자기 임계값, 10/7 사건)")
    ax.set_ylabel("Jaccard")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "분포_LOIO_일치도.png"), dpi=120); plt.close(fig)
    # 쌍별 히트맵 (TTC)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    im0 = axes[0].imshow(shift_t.to_numpy(dtype=float), cmap="YlOrRd")
    axes[0].set_title("쌍별 임계값 이동량 |q_i - q_j| (TTC, 초)")
    im1 = axes[1].imshow(jac_t.to_numpy(dtype=float), cmap="YlGnBu", vmin=0.5, vmax=1)
    axes[1].set_title("쌍별 판정 일치도 Jaccard (TTC)")
    for ax, im in ((axes[0], im0), (axes[1], im1)):
        ax.set_xticks(range(len(shift_t))); ax.set_xticklabels(shift_t.columns)
        ax.set_yticks(range(len(shift_t))); ax.set_yticklabels(shift_t.index)
        ax.set_xlabel("적용받는 교차로 j"); ax.set_ylabel("임계값 제공 교차로 i")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "히트맵_쌍별전이.png"), dpi=120); plt.close(fig)
    # H1 스윕
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(tau_df["threshold"], tau_df["kendall_tau_vs_1.5s"], color="#4C78A8", marker="o", ms=3)
    ax.axhline(0.8, color="crimson", ls=":", label="τ=0.8 (H1 관행 기준)")
    ax.set_xlabel("고정 TTC 임계값 (초)")
    ax.set_ylabel("순위 상관 (vs 1.5초 기준, Kendall τ)")
    ax.set_title("H1: 임계값 선택에 따른 교차로 위험 순위 안정성")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "분포_순위안정성.png"), dpi=120); plt.close(fig)


def main():
    out = "data/processed/transfer"
    os.makedirs(out, exist_ok=True)
    ttc, pet = th07.load_events("data/processed/conflicts")
    print(f"TTC {len(ttc):,} / PET {len(pet):,} (필터 후)", flush=True)

    lo = pd.concat([loio(ttc, "min_ttc", "TTC"), loio(pet, "pet_centroid", "PET")],
                   ignore_index=True)
    lo.to_csv(os.path.join(out, "loio.csv"), index=False, encoding="utf-8-sig")
    print("\n=== LOIO 20-fold ===")
    print(lo.round(3).to_string(index=False))

    shift_t, jac_t = pairwise(ttc, "min_ttc")
    shift_p, jac_p = pairwise(pet, "pet_centroid")
    shift_t.to_csv(os.path.join(out, "pairwise_shift_ttc.csv"), encoding="utf-8-sig")
    jac_t.to_csv(os.path.join(out, "pairwise_jaccard_ttc.csv"), encoding="utf-8-sig")
    shift_p.to_csv(os.path.join(out, "pairwise_shift_pet.csv"), encoding="utf-8-sig")
    jac_p.to_csv(os.path.join(out, "pairwise_jaccard_pet.csv"), encoding="utf-8-sig")
    print(f"\n쌍별(TTC): 이동량 중앙 {np.nanmedian(shift_t.values):.3f}s, "
          f"Jaccard 중앙 {np.nanmedian(jac_t.values):.3f}")

    daily = pd.read_csv("data/processed/intersection_daily.csv")
    expo = daily.groupby("intersection")["effective_min"].sum() / 60.0   # 시간
    rates, tau_df = sweep_ranks(ttc, expo)
    rates.to_csv(os.path.join(out, "sweep_rate_ttc.csv"), encoding="utf-8-sig")
    tau_df.to_csv(os.path.join(out, "sweep_tau_ttc.csv"), index=False, encoding="utf-8-sig")
    print(f"H1 스윕: 문헌범위(0.5~6.0s) 내 최소 Kendall τ = {tau_df['kendall_tau_vs_1.5s'].min():.3f}")

    nf = pd.read_csv("data/processed/thresholds/noise_floor.csv")
    plots(lo, shift_t, jac_t, tau_df, nf)
    print(f"\n저장: {out}/, 그림: outputs/transfer/")


if __name__ == "__main__":
    matplotlib.use("Agg")
    main()