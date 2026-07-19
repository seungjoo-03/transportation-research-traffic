# -*- coding: utf-8 -*-
"""data/raw의 궤적 ZIP을 data/interim에 푼다.

압축을 푸는 것 외에는 아무것도 하지 않는다. 컬럼을 빼거나 더하지 않고,
자료형도 바꾸지 않으며, 값도 그대로 둔다. interim은 원본의 충실한 사본이어야
나중에 "원본에 무엇이 있었는지"를 언제든 확인할 수 있다.

속도 재계산, 시간 변환, 이동류 분류 같은 가공은 전처리 단계(03)에서 한다.

풀린 CSV는 교차로-날짜별 폴더에 세션 10개씩 들어간다.
    data/interim/2022-10-04_G/2022-10-04_G_AM1.csv ...

이미 푼 파일(크기 일치)은 건너뛴다. 중단해도 다시 실행하면 이어서 진행된다.
원본 ZIP은 건드리지 않는다.

사용: python src/02_extract.py [입력=data/raw] [출력=data/interim]
"""
import os
import sys
import time
import zipfile


def extract_zip(zip_path: str, out_root: str) -> tuple[int, int, int]:
    """ZIP 하나를 풀고 (푼 파일 수, 건너뛴 수, 바이트) 반환"""
    stem = os.path.basename(zip_path)[:-4]           # 2022-10-04_G
    dest = os.path.join(out_root, stem)
    os.makedirs(dest, exist_ok=True)

    done = skipped = nbytes = 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            target = os.path.join(dest, os.path.basename(info.filename))
            if os.path.exists(target) and os.path.getsize(target) == info.file_size:
                skipped += 1
                nbytes += info.file_size
                continue
            with z.open(info) as src, open(target, "wb") as dst:
                while chunk := src.read(1 << 22):     # 4MB씩
                    dst.write(chunk)
            done += 1
            nbytes += info.file_size
    return done, skipped, nbytes


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/interim"
    os.makedirs(out, exist_ok=True)

    # 궤적 ZIP만 (날짜로 시작). segmentations·orthophotos 등 부속 파일은 제외
    zips = sorted(
        os.path.join(src, f) for f in os.listdir(src)
        if f.endswith(".zip") and f[0].isdigit()
    )
    total_raw = 0
    for zp in zips:
        with zipfile.ZipFile(zp) as z:
            total_raw += sum(i.file_size for i in z.infolist())
    print(f"궤적 ZIP {len(zips)}개 → {out}/  (풀면 약 {total_raw/1e9:.0f}GB)", flush=True)

    t0 = time.time()
    tot_done = tot_skip = tot_bytes = 0
    for i, zp in enumerate(zips, 1):
        d, s, b = extract_zip(zp, out)
        tot_done += d
        tot_skip += s
        tot_bytes += b
        print(
            f"[{i}/{len(zips)}] {os.path.basename(zp)[:-4]}  "
            f"푼 파일 {d}개, 건너뜀 {s}개  "
            f"(누적 {tot_bytes/1e9:.1f}GB, {time.time()-t0:.0f}초)",
            flush=True,
        )

    print(
        f"\n완료: 새로 푼 파일 {tot_done}개, 건너뛴 파일 {tot_skip}개, "
        f"총 {tot_bytes/1e9:.1f}GB, {time.time()-t0:.0f}초",
        flush=True,
    )


if __name__ == "__main__":
    main()