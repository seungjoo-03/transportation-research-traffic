# -*- coding: utf-8 -*-
"""Songdo Traffic v2 (Zenodo 17924857) 전체 다운로드.

궤적 ZIP 80개 + segmentations + orthophotos + master_frames + README/LICENSE를
data_raw/에 내려받는다 (sample_videos.zip 26.8GB는 분석에 불필요하므로 제외).
이미 받은 파일(크기 일치)은 건너뛰고, 완료 후 MD5를 검증한다. 중단돼도
재실행하면 이어서 진행된다.

사용: python src/01_download_data.py [대상 디렉토리=data_raw]
"""
import hashlib
import os
import sys
import time

import requests

RECORD_API = "https://zenodo.org/api/records/17924857"
EXCLUDE = {"sample_videos.zip"}
CHUNK = 1024 * 1024


def md5sum(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK * 8), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: str, size: int, retries: int = 5) -> None:
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                done = 0
                t0 = time.time()
                with open(dest + ".part", "wb") as f:
                    for chunk in r.iter_content(CHUNK):
                        f.write(chunk)
                        done += len(chunk)
                os.replace(dest + ".part", dest)
                mbps = done / 1e6 / max(time.time() - t0, 0.1)
                print(f"  받음 {done/1e6:,.1f}MB ({mbps:.1f}MB/s)", flush=True)
                return
        except Exception as e:
            print(f"  재시도 {attempt}/{retries}: {e}", flush=True)
            time.sleep(min(30, 5 * attempt))
    raise RuntimeError(f"다운로드 실패: {url}")


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else "data_raw"
    os.makedirs(target, exist_ok=True)

    rec = requests.get(RECORD_API, timeout=60).json()
    files = [f for f in rec["files"] if f["key"] not in EXCLUDE]
    files.sort(key=lambda f: f["size"])  # 작은 파일부터 — 초반에 빠른 진행 확인
    total_gb = sum(f["size"] for f in files) / 1e9
    print(f"대상 {len(files)}개 파일, 총 {total_gb:.2f}GB → {target}/", flush=True)

    failed = []
    for i, f in enumerate(files, 1):
        name, size = f["key"], f["size"]
        dest = os.path.join(target, name)
        if os.path.exists(dest) and os.path.getsize(dest) == size:
            print(f"[{i}/{len(files)}] {name} — 이미 있음, 건너뜀", flush=True)
            continue
        print(f"[{i}/{len(files)}] {name} ({size/1e6:,.1f}MB)", flush=True)
        url = f"{RECORD_API}/files/{name}/content"
        try:
            download(url, dest, size)
        except RuntimeError as e:
            print(f"  !! {e}", flush=True)
            failed.append(name)
            continue
        expected = f.get("checksum", "").replace("md5:", "")
        if expected and md5sum(dest) != expected:
            print(f"  !! MD5 불일치: {name} — 삭제, 다음 실행에서 재시도", flush=True)
            os.remove(dest)
            failed.append(name)

    if failed:
        print(f"\n실패 {len(failed)}건: {failed} — 스크립트를 재실행하면 이어받습니다", flush=True)
        sys.exit(1)
    print("\n전체 다운로드·검증 완료", flush=True)


if __name__ == "__main__":
    main()
