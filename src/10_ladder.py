# -*- coding: utf-8 -*-
"""보정 사다리 (RQ4·H4 세부): 현지 자료가 얼마나 있어야 판정이 회복되는가.

각 교차로 X에 대해, X의 훈련기간(10/4~06)에서 현지 자료를
  1세션(중앙 11분) → 3세션(~33분) → 1일(~110분) → 3일(전체)
씩 무작위로 뽑아 보정 임계값을 만들고, X의 시험일(10/7) 사건에 적용해
자기 임계값 대비 이동량·판정 일치도(Jaccard)를 잰다. 무작위 추출 반복(B=100)
으로 분포를 보고한다(계획서 §4: 어느 세션을 뽑느냐로 결과가 달라지므로).
비교선: 19곳 통합 임계값(무보정 전이, 08의 loio.csv).

출력: data/processed/transfer/ladder.csv
그림: outputs/transfer/분포_보정사다리.png  (x=현지 자료량, y=일치도 중앙·IQR)
사용: python src/10_ladder.py
"""
import importlib.util
import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("th07", os.path.join(_here, "07_thresholds.py"))
th07 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(th07)

P = th07.P_MAIN
TRAIN_DATES = ["2022-10-04", "2022-10-05", "2022-10-06"]
TEST_DATE = "2022-10-07"
B = 100
SEED = 20260723
LEVELS = [("1세션", "session", 1), ("3세션", "session", 3), ("1일", "day", 1), ("3일", "all", None)]


def q5(s):
    return s.quantile(P) if len(s) >= 20 else np.nan


def jaccard(vals, qa, qb):
    A = vals <= qa
    B_ = vals <= qb
    u = (A | B_).sum()
    return (A & B_).sum() / u if u else np.nan


def ladder_for(df, col, indicator):
    rng = np.random.default_rng(SEED)
    rows = []
    for X in sorted(df["intersection"].unique()):
        train = df[(df["intersection"] == X) & (df["date"].isin(TRAIN_DATES))]
        tv = df[(df["intersection"] == X) & (df["date"] == TEST_DATE)][col]
        q_own = q5(tv)
        if pd.isna(q_own):
            continue
        sess_keys = train["sess_key"].unique()
        dates = train["date"].unique()
        for name, unit, k in LEVELS:
            draws = 1 if unit == "all" else B
            for b in range(draws):
                if unit == "session":
                    if len(sess_keys) < k:
                        continue
                    pick = rng.choice(sess_keys, size=k, replace=False)
                    sub = train[train["sess_key"].isin(pick)][col]
                elif unit == "day":
                    pick = rng.choice(dates, size=1, replace=False)
                    sub = train[train["date"].isin(pick)][col]
                else:
                    sub = train[col]
                q_cal = q5(sub)
                if pd.isna(q_cal):
                    continue
                rows.append({"indicator": indicator, "intersection": X, "level": name,
                             "rep": b, "n_cal": len(sub),
                             "shift": abs(q_cal - q_own),
                             "jaccard": jaccard(tv, q_cal, q_own)})
    return pd.DataFrame(rows)


def main():
    out = "data/processed/transfer"
    ttc, pet = th07.load_events("data/processed/conflicts")
    lad = pd.concat([ladder_for(ttc, "min_ttc", "TTC"),
                     ladder_for(pet, "pet_centroid", "PET")], ignore_index=True)
    lad.to_csv(os.path.join(out, "ladder.csv"), index=False, encoding="utf-8-sig")

    lo = pd.read_csv(os.path.join(out, "loio.csv"))
    order = [l[0] for l in LEVELS]
    print("=== 보정 사다리 (교차로·반복 통합 중앙값) ===")
    summ = (lad.groupby(["indicator", "level"])
            .agg(shift_med=("shift", "median"), jac_med=("jaccard", "median"),
                 jac_q1=("jaccard", lambda s: s.quantile(0.25)),
                 jac_q3=("jaccard", lambda s: s.quantile(0.75)))
            .reset_index())
    print(summ.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, ind, col in ((axes[0], "TTC", "#4C78A8"), (axes[1], "PET", "#E45756")):
        d = summ[summ["indicator"] == ind].set_index("level").reindex(order)
        x = np.arange(len(order))
        ax.plot(x, d["jac_med"], marker="o", color=col, label="현지 보정(중앙)")
        ax.fill_between(x, d["jac_q1"], d["jac_q3"], color=col, alpha=0.2, label="IQR")
        base = lo[lo["indicator"] == ind]["jac_pooled"].median()
        ax.axhline(base, color="gray", ls="--", label=f"무보정 통합값 (중앙 {base:.2f})")
        ax.set_xticks(x); ax.set_xticklabels(order)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("현지 보정 자료량"); ax.set_ylabel("판정 일치도 (Jaccard)")
        ax.set_title(f"{ind}: 현지 자료량에 따른 판정 회복")
        ax.legend(loc="lower right")
    os.makedirs("outputs/transfer", exist_ok=True)
    fig.tight_layout(); fig.savefig("outputs/transfer/분포_보정사다리.png", dpi=120); plt.close(fig)
    print("\n저장: data/processed/transfer/ladder.csv, outputs/transfer/분포_보정사다리.png")


if __name__ == "__main__":
    matplotlib.use("Agg")
    main()