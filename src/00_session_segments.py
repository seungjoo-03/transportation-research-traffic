# -*- coding: utf-8 -*-
"""세션 내 호버링 세그먼트 검출·실측.

Songdo Traffic의 세션 CSV는 연속 관측이 아니라 드론 순회에 따른 2~4분 호버링
세그먼트의 묶음이다(Fonod et al., 2025). 이 스크립트는 세션별로 2초 초과
시간공백을 기준으로 세그먼트를 검출해 유효 관측시간(노출량 분모)과 세션
시각표를 산출한다. PET는 세그먼트 내부에서만 계산 가능하다(연구계획서 §2.2).

사용: python 00_session_segments.py <세션 CSV들이 있는 디렉토리> [...]
      디렉토리 예: data/songdo_sample/2022-10-04_G
"""
import glob
import os
import sys

import pandas as pd

GAP_THRESHOLD_S = 2.0  # 이 값 초과 공백 = 관측 중단(호버링 이동)
FPS = 30


def hms(sec: float) -> str:
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:04.1f}"


def analyze_session(csv_path: str) -> dict:
    df = pd.read_csv(csv_path, usecols=["Local_Time", "Vehicle_ID"])
    t = pd.to_timedelta(df["Local_Time"]).dt.total_seconds()
    ts = t.drop_duplicates().sort_values().to_numpy()

    segments = []
    seg_start = prev = ts[0]
    for cur in ts[1:]:
        if cur - prev > GAP_THRESHOLD_S:
            segments.append((seg_start, prev))
            seg_start = cur
        prev = cur
    segments.append((seg_start, prev))

    effective = sum(e - s for s, e in segments)
    return {
        "session": os.path.basename(csv_path).rsplit("_", 1)[-1].replace(".csv", ""),
        "start": hms(ts[0]),
        "end": hms(ts[-1]),
        "span_min": (ts[-1] - ts[0]) / 60,
        "effective_min": effective / 60,
        "n_segments": len(segments),
        "segments": "; ".join(f"{hms(s)}~{hms(e)} ({(e - s) / 60:.1f}분)" for s, e in segments),
        "rows": len(df),
        "vehicles": df["Vehicle_ID"].nunique(),
        "avg_concurrent": len(df) / (effective * FPS) if effective > 0 else float("nan"),
    }


def main(dirs: list[str]) -> None:
    for d in dirs:
        csvs = sorted(glob.glob(os.path.join(d, "*.csv")))
        if not csvs:
            print(f"[건너뜀] CSV 없음: {d}")
            continue
        rows = [analyze_session(f) for f in csvs]
        out = pd.DataFrame(rows)
        print(f"\n=== {os.path.basename(d)} ===")
        print(out.drop(columns="segments").to_string(index=False, float_format="%.2f"))
        for r in rows:
            print(f"  {r['session']}: {r['segments']}")
        total = out["effective_min"].sum()
        print(f"  유효 관측시간 합계: {total:.1f}분 (명목 {out['span_min'].sum():.1f}분)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
