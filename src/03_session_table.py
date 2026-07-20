# -*- coding: utf-8 -*-
"""20개 교차로 전체의 세션 구조를 실측해 표 두 개로 저장한다.

Songdo Traffic 세션 CSV는 연속 관측이 아니라 드론 호버링 구간의 묶음이다
(Fonod et al., 2025). 파일의 시간 범위는 약 15분이지만 실제로 기록이 있는
시간은 그보다 짧다. 2초를 넘는 시간 공백을 세그먼트 경계로 보고, 세그먼트
길이의 합을 그 세션의 유효 관측시간으로 잡는다.

이 유효 관측시간이 상충률의 분모(노출량)가 되므로 전 세션을 실측한다.
계획서 §8의 "유효 3~4분, 하루 약 35분"을 이 결과로 확정한다.

출력 (data/processed/):
  session_segments.csv   세션 800개 한 줄씩 (유효시간·조각수·행수·차량수)
  intersection_daily.csv 교차로-날짜 80개 한 줄씩 (하루 유효시간 합계)

사용: python src/03_session_table.py [입력=data/interim] [출력=data/processed]
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

GAP_THRESHOLD_S = 2.0   # 이 값을 넘는 공백 = 관측 중단(드론 이동)
FPS = 30.0


def local_time_to_sec(series: pd.Series) -> pd.Series:
    """'17:11:40.561' → 61900.561 (자정 기준 초). 벡터화."""
    parts = series.str.split(":", expand=True)
    return (
        parts[0].astype("int32") * 3600
        + parts[1].astype("int32") * 60
        + parts[2].astype("float64")
    )


def analyze_session(csv_path: str) -> dict:
    df = pd.read_csv(csv_path, usecols=["Local_Time", "Vehicle_ID"], low_memory=False)
    ts = np.sort(local_time_to_sec(df["Local_Time"]).unique())

    # 2초 초과 공백에서 끊어 세그먼트로 나눈다
    gaps = np.diff(ts)
    cut = np.where(gaps > GAP_THRESHOLD_S)[0]
    starts = np.concatenate(([0], cut + 1))
    ends = np.concatenate((cut, [len(ts) - 1]))
    seg_lengths = ts[ends] - ts[starts]        # 각 세그먼트 길이(초)

    stem = os.path.basename(csv_path)[:-4]      # 2022-10-04_G_PM1
    date, inter, session = stem.rsplit("_", 2)  # 2022-10-04 / G / PM1
    return {
        "date": date,
        "intersection": inter,
        "session": session,
        "period": session[:2],                  # AM / PM
        "start": ts[0],
        "end": ts[-1],
        "span_min": (ts[-1] - ts[0]) / 60,
        "effective_min": seg_lengths.sum() / 60,
        "n_segments": len(seg_lengths),
        "seg_detail_s": ";".join(f"{s:.0f}" for s in seg_lengths),
        "rows": len(df),
        "vehicles": df["Vehicle_ID"].nunique(),
    }


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "data/interim"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/processed"
    os.makedirs(out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(src, "*", "*.csv")))
    print(f"세션 파일 {len(files)}개 실측 시작", flush=True)

    rows = []
    for i, f in enumerate(files, 1):
        rows.append(analyze_session(f))
        if i % 100 == 0 or i == len(files):
            print(f"  {i}/{len(files)}", flush=True)

    sess = pd.DataFrame(rows).sort_values(["intersection", "date", "session"])
    sess.to_csv(os.path.join(out, "session_segments.csv"), index=False, encoding="utf-8-sig")

    # 교차로-날짜 단위 요약
    daily = (
        sess.groupby(["intersection", "date"])
        .agg(
            sessions=("session", "count"),
            effective_min=("effective_min", "sum"),
            span_min=("span_min", "sum"),
            rows=("rows", "sum"),
        )
        .reset_index()
    )
    daily.to_csv(os.path.join(out, "intersection_daily.csv"), index=False, encoding="utf-8-sig")

    # 요약 출력
    print("\n=== 전체 요약 ===")
    print(f"세션당 유효 관측시간: 중앙값 {sess.effective_min.median():.2f}분, "
          f"범위 {sess.effective_min.min():.2f}~{sess.effective_min.max():.2f}분")
    print(f"세션당 세그먼트 수: 중앙값 {sess.n_segments.median():.0f}, "
          f"범위 {sess.n_segments.min()}~{sess.n_segments.max()}")
    print(f"교차로-하루 유효 관측시간: 중앙값 {daily.effective_min.median():.1f}분, "
          f"범위 {daily.effective_min.min():.1f}~{daily.effective_min.max():.1f}분")
    print("\n첫 회차(AM1/PM1)만 짧은지 확인:")
    by_session = sess.groupby("session").effective_min.median().reindex(
        ["AM1", "AM2", "AM3", "AM4", "AM5", "PM1", "PM2", "PM3", "PM4", "PM5"]
    )
    print(by_session.to_string(float_format="%.2f"))
    print(f"\n저장: {out}/session_segments.csv, {out}/intersection_daily.csv")


if __name__ == "__main__":
    main()