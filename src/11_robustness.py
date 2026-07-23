# -*- coding: utf-8 -*-
"""민감도 일괄 + forward-chaining (분석 4단계): 결론이 선택값에 걸려 있지 않은가.

① 민감도: 사전지정 파라미터를 변형해 핵심 판정(H3 잡음초과 곳 수, H4 오차 중앙·p)이
   유지되는지 표로 제시. 변형 축:
   - 인공물 필터: TTC 최소거리 {0.5, 1.0, 1.5}m / PET 하한 {0.1, 0.2, 0.5}s
   - 분위수 p: {1%, 3%, 5%, 10%}
   - 좌표오차 교차로 L·P 제외
   - PET 길이보정값 사용(중심점 대신)
   잡음 바닥은 변형별 재계산(B=50 — 판정 재현용 축소, 주분석은 07의 B=200).
   ※ ε·지속샘플 변형은 06 재계산이 필요해 본 표에서 제외(별도 계획 명시).
② forward-chaining 3회(계획서 §4 '익일 재현성'): 각 교차로 자기 임계값이
   훈련(과거)→시험(다음날)으로 재현되는가. 10/4→10/5, 10/4~5→10/6, 10/4~6→10/7.

출력: data/processed/stats/robustness.csv, forward_chaining.csv
사용: python src/11_robustness.py
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

P_MAIN = 0.05
AZ_MIN = 30.0
B_NF = 50
SEED = 20260723
TRAIN_DATES = ["2022-10-04", "2022-10-05", "2022-10-06"]
TEST_DATE = "2022-10-07"


def q_at(s, p):
    return s.quantile(p) if len(s) >= 20 else np.nan


def jaccard(vals, qa, qb):
    A = vals <= qa
    B = vals <= qb
    u = (A | B).sum()
    return (A & B).sum() / u if u else np.nan


def noise_p95(df, col, p, rng):
    """교차로별 반분할(B_NF회) 이동량 p95 — 변형별 잡음 바닥."""
    out = {}
    for inter, sub in df.groupby("intersection"):
        sess = np.asarray(sub["sess_key"].unique(), dtype=object)   # Arrow→numpy (셔플 중복 방지)
        if len(sess) < 4:
            continue
        shifts = []
        for _ in range(B_NF):
            rng.shuffle(sess)
            h = len(sess) // 2
            A = sub[sub["sess_key"].isin(sess[:h])][col]
            Bv = sub[sub["sess_key"].isin(sess[h:])][col]
            qa, qb = q_at(A, p), q_at(Bv, p)
            if pd.notna(qa) and pd.notna(qb):
                shifts.append(abs(qa - qb))
        if shifts:
            out[inter] = float(np.quantile(shifts, 0.95))
    return out


def loio_metrics(df, col, p):
    """LOIO: 교차로별 (통합 이동량, 현지 이동량). H3·H4 판정 재료."""
    train = df[df["date"].isin(TRAIN_DATES)]
    test = df[df["date"] == TEST_DATE]
    q_train = train.groupby("intersection")[col].apply(lambda s: q_at(s, p))
    rows = {}
    for X in q_train.index:
        pooled = q_train.drop(index=X).median()
        local = q_train[X]
        tv = test.loc[test["intersection"] == X, col]
        own = q_at(tv, p)
        if pd.isna(own) or pd.isna(local):
            continue
        rows[X] = (abs(pooled - own), abs(local - own))
    return rows


def evaluate(name, ttc, pet, p=P_MAIN, pet_col="pet_centroid"):
    """한 변형에 대한 핵심 판정 요약 한 줄."""
    rng = np.random.default_rng(SEED)
    res = {"variant": name, "p": p}
    for ind, df, col in (("TTC", ttc, "min_ttc"), ("PET", pet, pet_col)):
        lm = loio_metrics(df, col, p)
        nf = noise_p95(df, col, p, rng)
        common = [i for i in lm if i in nf]
        exceed = sum(lm[i][0] > nf[i] for i in common)
        pooled = [lm[i][0] for i in common]
        local = [lm[i][1] for i in common]
        try:
            pval = wilcoxon(local, pooled, alternative="less").pvalue
        except ValueError:
            pval = np.nan
        res[f"{ind}_h3"] = f"{exceed}/{len(common)}"
        res[f"{ind}_h3_supported"] = bool(exceed > len(common) / 2)
        res[f"{ind}_h4_med"] = f"{np.median(local):.3f}<{np.median(pooled):.3f}"
        res[f"{ind}_h4_p"] = round(float(pval), 4)
    return res


def forward_chaining(ttc, pet):
    """익일 재현성: 자기 임계값(과거 훈련창)이 다음날 자기 값과 얼마나 어긋나나."""
    splits = [(["2022-10-04"], "2022-10-05"),
              (["2022-10-04", "2022-10-05"], "2022-10-06"),
              (TRAIN_DATES, TEST_DATE)]
    rows = []
    for ind, df, col in (("TTC", ttc, "min_ttc"), ("PET", pet, "pet_centroid")):
        for tr, te in splits:
            for X, sub in df.groupby("intersection"):
                qa = q_at(sub[sub["date"].isin(tr)][col], P_MAIN)
                tv = sub[sub["date"] == te][col]
                qb = q_at(tv, P_MAIN)
                if pd.isna(qa) or pd.isna(qb):
                    continue
                rows.append({"indicator": ind, "train": "+".join(d[-2:] for d in tr),
                             "test": te[-2:], "intersection": X,
                             "shift": abs(qa - qb), "jaccard": jaccard(tv, qa, qb)})
    return pd.DataFrame(rows)


def main():
    out = "data/processed/stats"
    os.makedirs(out, exist_ok=True)
    base = "data/processed/conflicts"
    ttc0 = pd.read_csv(os.path.join(base, "ttc_events.csv"))
    pet0 = pd.read_csv(os.path.join(base, "pet_events.csv"))
    for d in (ttc0, pet0):
        d["sess_key"] = d["date"] + "_" + d["session"]
    pet0 = pet0[pet0["azdiff"] >= AZ_MIN]

    def F(td, pm):     # 인공물 필터 적용
        return (ttc0[ttc0["dist_min"] >= td].copy(),
                pet0[pet0["pet_centroid"] >= pm].copy())

    rows = []
    ttc_m, pet_m = F(1.0, 0.2)
    rows.append(evaluate("주분석(1.0m/0.2s/p5)", ttc_m, pet_m))
    rows.append(evaluate("TTC필터 0.5m", *F(0.5, 0.2)))
    rows.append(evaluate("TTC필터 1.5m", *F(1.5, 0.2)))
    rows.append(evaluate("PET필터 0.1s", *F(1.0, 0.1)))
    rows.append(evaluate("PET필터 0.5s", *F(1.0, 0.5)))
    for pv in (0.01, 0.03, 0.10):
        rows.append(evaluate(f"p={int(pv*100)}%", ttc_m, pet_m, p=pv))
    keepLP = ~ttc_m["intersection"].isin(["L", "P"])
    keepLP_p = ~pet_m["intersection"].isin(["L", "P"])
    rows.append(evaluate("L·P 제외(좌표오차)", ttc_m[keepLP], pet_m[keepLP_p]))
    petc = pet_m.dropna(subset=["pet_corrected"])
    petc = petc[petc["pet_corrected"] >= 0]
    rows.append(evaluate("PET 길이보정값", ttc_m, petc, pet_col="pet_corrected"))

    rob = pd.DataFrame(rows)
    rob.to_csv(os.path.join(out, "robustness.csv"), index=False, encoding="utf-8-sig")
    print("=== 민감도 일괄 (핵심 판정 유지 여부) ===")
    print(rob[["variant", "TTC_h3", "TTC_h3_supported", "TTC_h4_p",
               "PET_h3", "PET_h3_supported", "PET_h4_p"]].to_string(index=False))

    fc = forward_chaining(ttc_m, pet_m)
    fc.to_csv(os.path.join(out, "forward_chaining.csv"), index=False, encoding="utf-8-sig")
    print("\n=== forward-chaining 익일 재현성 (교차로 중앙값) ===")
    print(fc.groupby(["indicator", "train", "test"])
            .agg(shift_med=("shift", "median"), jac_med=("jaccard", "median"))
            .round(3).to_string())
    print(f"\n저장: {out}/robustness.csv, forward_chaining.csv")


if __name__ == "__main__":
    main()