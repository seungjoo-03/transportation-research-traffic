# 교통상충 TTC·PET 임계값 안정성 연구 (transportation-research-traffic)

인천 송도 20개 교차로 드론 궤적 데이터(Songdo Traffic)를 이용해, 고정 TTC·PET 임계값에 따른 교통상충 판정의 **교차로 간 안정성(공간 전이성)과 익일 재현성**을 검증하는 연구입니다.

- 연구계획서: [연구계획_교통상충_TTC_PET_임계값_전이성.md](연구계획_교통상충_TTC_PET_임계값_전이성.md) (핵심 주제·연구질문·가설)
- 방법·데이터 상세: [연구방법_데이터_상세.md](연구방법_데이터_상세.md) (데이터 사양·검증설계·산출·통계·일정)
- 풀페이퍼 초안 목표: 2026-07-24 (방법·데이터 절 완결 + 5~10개 교차로 본분석 탑재 기준)

## 데이터 준비

원자료는 용량 문제로 저장소에 포함하지 않습니다. 아래에서 직접 내려받아 `data/` 아래에 배치하세요.

- **Songdo Traffic v2** (CC BY 4.0): https://zenodo.org/records/17924857
  - 필요 파일: 궤적 ZIP 80개(총 12.21GB), `segmentations.zip`(차로 폴리곤), `orthophotos.zip`(1.8GB)
  - `sample_videos.zip`(26.8GB)은 분석에 불필요
- 데이터 논문: Fonod, R., Cho, H., Yeo, H., & Geroliminis, N. (2025). *Transportation Research Part C, 178*, 105205. https://doi.org/10.1016/j.trc.2025.105205

참고문헌 PDF도 저작권 문제로 저장소에서 제외합니다. 서지정보는 연구계획서 §17을 참조하세요.

## 폴더 구조

```text
├─ data/                  # (git 제외) 원자료·표본
├─ references/            # (PDF는 git 제외) 핵심 논문
├─ src/                   # 분석 파이프라인 (연구계획서 §10 참조)
│  └─ 00_session_segments.py   # 세션 내 호버링 세그먼트 검출·실측
├─ outputs/               # 그림·표
└─ 연구계획_교통상충_TTC_PET_임계값_전이성.md
```

## 핵심 설계 요약

- TTC는 추종·동일경로 상충 전용, PET는 교차·회전 상충 전용 (신호현시 부재 대응)
- 세션은 연속 관측이 아닌 2~4분 호버링 세그먼트의 묶음 — 노출량은 세그먼트 실측 관측시간 기준
- 검증: leave-one-intersection-out(20-fold) + forward-chaining 시간검증(3회)
- 사전 지정 주분석: TTC 1.5s, PET 2.0s, 분위수 p=5% (다중비교 통제)
