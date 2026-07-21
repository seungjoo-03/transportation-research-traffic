# TRIP 논문 정독 정리 노트 — `06_conflicts.py` TTC/PET 상충 사건표 구현용

> 출처: Espadaler-Clapés et al. (2025), *Transportation Research Interdisciplinary Perspectives* 32:101490 (TRIP). 5측면(ttc / pet / classify / kinematics / thresholds) 정독·검증 JSON 기반.
> 검증 결과: 인용된 핵심 주장은 **전부 `confirmed`**. `refuted`/`uncertain` 판정은 없었음. 단 **Eq.(1)(2)(3)이 PDF 추출에서 줄바꿈·아래첨자 깨짐** → 맥락으로 복원(§7에 재대조 목록). 신뢰도 `medium` 항목은 본문에 별도 표기.

---

## 0. 전체 파이프라인 (SSM 3단계) — 우리 06의 뼈대

> 원문(p.5): *"we employ well-known SSM to quantify unsafe traffic events... 1) detecting vehicle interactions, 2) identifying the type of interaction and 3) quantifying unsafe events through SSM."*

1. **근접쌍 탐지** — 속도기반 동적 탐색공간(부채꼴)으로 후보 surrounding 수집 (§4)
2. **유형 분류** — 3각도(θv, θp, θc) + 거리감소 / 방위각차로 rear-end vs crossing 분리 (§3)
3. **SSM 산출** — rear-end→TTC(연속), crossing→PET(단일 이벤트) (§1, §2)

→ `06_conflicts.py`는 이 3단계를 각각 함수로 분리하는 게 원문과 정합.

---

## 1. TTC

### 1.1 수식 (Eq.2, p.5) — ⚠️ PDF 깨짐, 맥락 복원

원문 raw(깨진 상태): `TTC = dist(sub, surr) / vsub −vsurr / ifvsub > vsurr / (2)`

**복원식:**
$$\text{TTC} = \frac{\text{dist(sub, surr)}}{v_{sub} - v_{surr}}, \quad \text{단 } v_{sub} > v_{surr}$$

- `dist(sub, surr)` = subject(뒤차)·surrounding(앞차) **중심점 사이 유클리드 거리** (둘 다 점 표현)
- `v_sub`, `v_surr` = 각 차량 **속력(스칼라)** — rear-end라 방향이 거의 같다는 가정
- **조건**: `v_sub > v_surr` (뒤차가 더 빨라 접근 중 = 분모 양수). 아니면 정의 안 함(계산 스킵)

> 개념(p.5): *"The TTC (Hayward, 1972) is the time required for two vehicles to collide if they continue at their present speed and along the same path. The TTC is a continuous parameter and can be calculated at any timestamp as long as the two road users are on a collision course."*

### 1.2 적용조건 — rear-end + 거리감소만

> (p.5-6): *"The TTC is primarily used for rear-end interactions... we only consider rear-end interactions, where the distance between the two road users is decreasing, ensuring that these interactions are always on a collision path. Consequently, this allows for a continuous computation of the TTC."*

**구현 필터 순서**: (1) 추종쌍 판정(θv<30° AND θp<30°) → (2) 거리 감소 `d(t) < d(t−1)` → (3) `v_sub > v_surr` → 매 프레임 TTC 산출. 세 조건은 서로 보강적(거리감소 ⇔ 대략 v_sub>v_surr).

### 1.3 차량 표현 — 점(중심점), 길이 미반영 ⚠️ 한계

> (p.4): *"we choose to represent each vehicle as a singular point, pinpointing the center of the bounding box at each timestep."*
> (p.5): *"the equation only considers the distance between the subject and surrounding vehicle (represented as single points) and does not include the length of the vehicle (bounding boxes... are axis-aligned with the stabilized video, hindering the accurate estimation of vehicle dimensions)."*

- 중심-중심 거리는 실제 범퍼 간극보다 `(L_앞/2 + L_뒤/2)`만큼 큼 → **TTC 약간 과대추정**. 한계 각주로 명시.
- 향후과제(p.11): *"the introduction of oriented bounding boxes should further improve the accuracy and reliability of the estimated safety metrics."*

### 1.4 그들이 쓴 임계값

- **새 임계값 유도 안 함.** 문헌 통상범위만 인용(p.3): *"typical ranges for TTC thresholds are between 1.5 and 4 s."*
- **실제 운영값 = 구간(bin)**: `0–2 s`(가장 위험) / `2–4 s`. Fig.8 캡션: *"a) junction #1 between 0 and 2 s, b)... between 2 and 4 s"*
- **관계분석(회귀)에는 `TTC < 4 s` 누적 컷오프** (p.10): *"link them to the observed TTC events under 4 s − since dangerous PET events are rare and TTC is a continuous parameter, TTC is a better metric in this case."*
- 정성 검증 예시(p.8-9, **medium**): 정지차에 접근하며 TTC가 볼록(convex) 진화(Fig.11).

### 1.5 집계 단위 — vehicle-frames / 시간

> (p.6): *"...according to TTC values (measured in interactions-frames per recorded hour as TTC is a continuous metric)."* Fig.8 캡션: *"Number of TTC events (vehicle-frames)."*

→ 임계값 미만 **프레임 수**를 세어 녹화시간(h)으로 정규화. (PET와 집계 단위 다름 주의)

---

## 2. PET

### 2.1 수식 (Eq.3, p.5) — ⚠️ PDF 깨짐, 맥락 복원

원문 raw(깨진 상태): `PET = t2 / c −t1 / c / (3)`

**복원식:**
$$\text{PET} = t^{2}_{c} - t^{1}_{c}$$

- 위첨자 1,2 = 두 차량, 아래첨자 c = **공통점 통과(crosses the common point)**
- `t^x_c` = 차량 x가 공통점을 통과한 **실측 시각** (예측 아님)
- 구현은 **절댓값** `|t2_c − t1_c|`로.

> (p.5): *"The PET (Allen et al., 1978) is a single value that can be directly measured from the trajectories and is the time difference between the two vehicles passing a common spatial zone but at different times and thus avoiding a collision (Eq. (3), where tx c is the time vehicle x crosses the common point)."*

### 2.2 공통점 정의 — 두 실제 궤적의 교차

> (p.5): *"the PET value is a single value that concerns the crossing of two actual trajectories."*

- **예측/투영 경로 아님.** 두 차량 실제 궤적 폴리라인의 XY 교차점(common spatial zone/point)을 찾고, 각 차량이 그 점을 지난 시각을 보간해 차이 산출.
- 원문은 산문에서 `common spatial zone`(공간영역)과 수식설명에서 `common point`(공통점)를 혼용.

### 2.3 적용조건 — 교차/측면(side)만, 방위각차 ≥30°

> (p.5): *"We impose a minimum of 30◦ azimuth difference in the crossing point to focus on side interactions."*

- 교차점에서 두 차량 azimuth 차 `|Δazimuth| ≥ 30°`일 때만 PET로 인정. rear-end(방위차<30°대)와 **상보적**.
- PET는 crossing 전용, rear-end는 TTC 전용 (§3.3).
- 예시(p.8): 직진차가 교차점 통과 직후 우회전 대기차가 가속해 2초 이내 공통점 통과.

### 2.4 임계값 — 명시적 정당화 없음 ⚠️

- **PET 전용 임계 근거 없음.** 1.5–4 s는 TTC 범위이며 PET용 아님(§5.1).
- 운영값: `0–2 s`(가장 위험) / `2–4 s` 구간 (Fig.9). 이동쌍 랭킹은 `PET < 4 s` 카운트(Fig.12).
- 희소성(p.8): *"the areas where PET under 2 s are detected present very low frequency of risky events (less than 1 event per hour in most cases)."* → 표본 수 병기 권장.

### 2.5 집계 단위 — 이벤트 수 / 시간

교차 이벤트 **1건 = 1카운트**(프레임 아님). 시간당 개수로 정규화.

---

## 3. 추종(rear-end) vs 교차(crossing) 판정 — SSM 매핑

### 3.1 세 각도 정의 (Saunier 2010 / Beauchamp 2022, p.5)

| 각도 | 정의 | 의미 |
|---|---|---|
| **θv** (speed angle) | 두 차량 속도벡터 사이 각 | 같은 방향 추종 여부 |
| **θp** (parallelism) | 거리벡터 vs (v_sub + v_surr) 합벡터 사이 각 | 평행 여부 |
| **θc** (collision) | 거리벡터 vs (v_sub − v_surr) 속도차벡터 사이 각 | 충돌경로 여부 |

각도는 `np.arctan2` 기반 벡터각, `0~180°`로 정규화. **rear-end 판정엔 θv, θp만 사용**(θc는 개념적 충돌경로 지표).

### 3.2 판정 임계

- **rear-end**: `θv < 30° AND θp < 30°` (p.5: *"the rear-end interactions are identified with the first two as follows: θv< 30◦ and θp < 30◦"*)
- **crossing/side**: 교차점에서 `|Δazimuth| ≥ 30°` (§2.3)

### 3.3 SSM 역할 분담 (핵심)

> (p.5): *"The TTC is primarily used for rear-end interactions... Finally, the PET value is a single value that concerns the crossing of two actual trajectories."*

- **rear-end(거리 감소, collision course) → TTC** (연속 시계열)
- **crossing/side(궤적 실제 교차, 방위차≥30°) → PET** (단일 이벤트)
- 안전분석 대상은 **collision 또는 crossing course인 것만**. non-crossing(멀어지는/안 만나는)은 버림.
  > (p.5): *"only interactions that are on a collision or crossing courses... are relevant for a safety analysis."*
- 두 지표를 **모든 pair에 무분별 적용 금지** — 유형 분류 후 각각의 후보군에만.

---

## 4. 궤적·kinematics 처리

### 4.1 프레임레이트 & 다운샘플

- 원자료: DJI MINI3 4K, 120m bird-eye, **29.97 FPS = 29.97 Hz**(네이티브), 위치도 29.97 Hz 기록.
- **상호작용/SSM 계산 단계에서는 5 Hz로 다운샘플**(Δt=0.2s). (p.5: *"we undersample the data to 5 Hz"*)
- → 06에서 상호작용 진입 전 `29.97Hz → 5Hz`(매 6프레임) 리샘플 명시.

### 4.2 좌표계

> (p.4): 픽셀 → GCP+호모그래피 → **투영좌표(UTM)** (필요시 WGS84).
> 절차: ①QGIS Georeferencer로 **첫 번째 안정화 영상의 첫 프레임** GCP 추출(=georeferenced frame, 유일 기준) → ②각 영상 첫 프레임을 georeferenced frame에 정합(H_frame) → ③GCP 호모그래피(H_GCP)로 raw projected.
> **최종 좌표 = H_GCP · H_frame · (영상 픽셀).** GCP는 한 번만, 나머지 영상은 프레임 정합만 매번.

- TTC/PET 거리·속도 계산은 **UTM(미터)** 에서. WGS84 쓰면 반드시 미터 투영 후 거리 계산.
- 우리 04가 이미 정사영상/미터 좌표를 다루므로 정합적.

### 4.3 속도 (Kalman + SG)

> (p.4): *"a coordinate-decoupled linear Kalman filter... white-noise acceleration model (Li and Jilkov, 2003)... Following (Mahajan et al., 2023)... we also use a Savitzky-Golay (SG) filter to reduce any unwanted noise."*

- 축분리 선형 칼만 + SG 이중 평활. **06의 v_sub, v_surr는 이 평활 속도를 그대로 사용.**
- 우리 `04_speed.py` 산출물이 SG 평활을 포함하는지 확인 → 없으면 06 이전에 SG(홀수 window, poly 2~3) 적용 검토.

### 4.4 방향 (azimuth)

> (p.4): *"the angle between the north and the vehicle direction measured clockwise... calculated as the angle created by the difference in coordinates... Due to the high frequency (29.97 Hz) we use a 2-second moving average to guarantee that the direction... is stable, especially at low speeds."*

- `azimuth = atan2(ΔEast, ΔNorth)`, `[0,360)` 정규화(북=0, 시계방향).
- **2초 이동평균 필수**(29.97Hz≈60프레임, 5Hz≈10샘플). 저속·정지 구간 노이즈 → 평활 없으면 상호작용 오판정.
- θv 계산과 PET 30° 교차조건에 직결. 우리 04/05 방위각 정의(북기준·시계방향·2초 평활)가 TRIP과 일치하는지 확인 후 재사용.

### 4.5 근접쌍 생성 (동적 탐색공간, Eq.1) — ⚠️ PDF 깨짐, 복원

원문 raw(깨진 상태): `r(v) = 5v 3.6, α(v) = min(100 + 0.8*v, 180) (1)`

**복원식** (v는 km/h):
$$r(v) = \frac{5v}{3.6}\ [\text{m}], \quad \alpha(v) = \min(100 + 0.8v,\ 180)\ [\text{deg}]$$

- `r` = 현재 속도로 **5초 주행거리**. 예: v=50 → r≈69.4m, α=140°. v=0 → r=0, α=100°.
- **두 파라미터는 subject 속도에만 의존** → 고속 차량은 정지차도 탐지.
- ⚠️ 정정: **α는 '비례'가 아니라 절편 100°·상한 180°의 아핀 함수**(원문 표현도 `increase linearly`). r만 순수 비례.
- 부채꼴 = subject azimuth ±(α/2), 반경 r 이내. (※ α를 '전체 부채꼴 각'으로 해석 — §7 대조 필요)
- 정지(v=0)면 r=0 → 후보 거의 없음(계산량 급감). 대안: KDTree/격자로 반경 후보 뽑고 각도 필터 후처리(동일 결과).
- 다층 구조(교량/고가): map-matching으로 같은 레벨끼리만. **송도가 단일 평면이면 생략 가능.**

### 4.6 timestep 스키마 (06 최소 입력)

> (p.4-5): *"every trajectory is described at each timestep with georeferenced pixel coordinates, projected coordinates (UTM), speed, acceleration, and azimuth. Each trajectory is identified with the original video file, a unique track ID and the type of vehicle."*

**체크리스트**: `track_id`, `t(시간)`, `x/y(UTM m)`, `speed`, `azimuth` (+선택 `acceleration`, `vehicle_type`).

### 4.7 정지·부분궤적·이상치

- CV/추적(BoT-SORT + 강화 칼만)이 일시 가림(나무·기둥) 처리, 궤적 파편화 감소 → **상류에서 이미 처리**되었다고 가정.
- ⚠️ junction#2는 ReID 부재 + 교량 가림으로 한 차량이 다른 ID로 분절 → OD 추출 곤란(단일평면 junction#1은 OD 추출 가능). **송도에 입체 가림 없으면 경미.**
- 노이즈·이상치 표준: 칼만 + SG (Mahajan 2023). 06에서도 최소 궤적 길이/물리적 속도 상한 방어필터 권장.

---

## 5. 임계값 한계 서술 + 상충 집계·공간분석

### 5.1 임계값에 대한 그들의 태도 (우리 논문 정당화 논리와 직결)

> (p.3, **medium**): *"the choice of a threshold to distinguish risky interactions is also a hot topic, and no solid conclusion has been reached in the literature (Papazikou et al., 2019). For example, typical ranges for TTC thresholds are between 1.5 and 4 s..."*

- **단일 컷오프 정당화 안 함.** 대신 구간(bin)으로 제시 = 사실상 다중 임계값/민감도.
- ⚠️ **인용범위(1.5–4s)와 실제 적용 하한(2s)이 불일치** — 1.5–4s는 오직 TTC용 인용, 실제 bin 하한은 2s.
- ⚠️ **두-구간(0–2s / 2–4s)은 공간분포(Fig.8/9)에만 국한.** 충돌쌍 랭킹(Fig.12)과 관계분석(Fig.13)은 **단일 누적 `<4s`** 사용. TTC 산식 자체엔 임계값 없음(집계 단계에서만 적용).
- **우리 차별점**: TRIP은 임계값 정당화를 회피 → 우리는 `p=5%` 잡음바닥 기반 단일 컷오프를 정당화(전이성 강건성 서술에 *"임계값을 올려도 동일 위험구역 재확인"* 논리 차용 가능).

### 5.2 집계 이원화

| 지표 | 집계 단위 | 근거 |
|---|---|---|
| **TTC** | 임계미만 **프레임 수** / 시간 (vehicle-frames/h) | 연속량 (p.6, Fig.8) |
| **PET** | **이벤트 수** / 시간 (교차 1건=1카운트) | 단일값 (p.7-8, Fig.9) |

### 5.3 공간분석·원인 해석

- **이벤트 수 히트맵**, 교차로 간 비교 시 **동일 legend scale 고정**(p.6-7). 특정 차로/정지선/중앙부/합류점 지목.
- 06 출력: 격자(예: 5m 셀)별 상충 카운트 남겨 히트맵화.
- **신호 위상 결합**(p.7): *"the low PET values occur mainly during the all-red phase and the transition of the green phase between conflicting movements."* → 위상 타임스탬프 조인 시 원인 해석 강화(송도 신호 확보 가능하면).
- **이동쌍(movement pair) 랭킹**(p.8, Fig.12): OD 결합 후 `PET<4s` 카운트로 상위 3쌍 지목 — 전부 우회전과 일치(*"which coincide with the right turns in each direction"*). → 각 이벤트에 자차·상대차 이동(OD leg 조합) 라벨 부착 필요.
- **관계분석 집계 해상도**(p.10): junction#1을 `5m×5m 셀 × 30초`로 집계, 셀 내 평균·표준편차 속도 vs `TTC<4s` 이벤트, **99퍼센타일 상한 포락선 + LOWESS**.

### 5.4 명시된 한계

- **차종 미분류**(p.6): *"we do not cluster the safety metrics with vehicle categories as the available samples for minority classes like buses or trucks are under-represented."* → 06도 전 차종 통합 집계(라벨은 유지, 군집 기준으론 미사용).
- **근접충돌의 확률적 특성**(p.11): *"the stochastic nature of near crashes should be highlighted: while an upper envelope curve has been found... further data and other metrics should be employed to find a clearer connection."* → 과대해석 경계, 산점 이질성·표본기간(하루치 single-day 한계) 명시.

---

## 6. 우리 06 구현 체크리스트 — TRIP 그대로 vs 조정

### ✅ TRIP 그대로 채택

| 항목 | 내용 | TRIP 근거 |
|---|---|---|
| TTC 산식 | `dist(중심,중심)/(v_sub−v_surr)`, `v_sub>v_surr`만 | Eq.2, p.5 |
| PET 산식 | `|t2_c − t1_c|`, 실제 궤적 교차점 | Eq.3, p.5 |
| rear-end 판정 | `θv<30° AND θp<30°` + 거리감소 | p.5 |
| crossing 판정 | 교차점 `|Δazimuth|≥30°` | p.5 |
| SSM 매핑 | rear-end→TTC, crossing→PET | p.5 |
| 근접쌍 | 부채꼴 `r=5v/3.6`, `α=min(100+0.8v,180)` | Eq.1, p.5 |
| 5Hz 다운샘플 | 상호작용 계산 전 리샘플 | p.5 |
| azimuth | 북기준 시계방향 + 2초 이동평균 | p.4 |
| 임계 bin | `0–2s`(critical)/`2–4s`, 랭킹은 `<4s` | Fig.8/9/12, p.6-10 |
| 집계 이원화 | TTC=vehicle-frames/h, PET=events/h | p.6-8 |
| 히트맵 | 동일 컬러스케일, 격자 셀 카운트 | p.6-7 |
| 차종 미군집 | 전 차종 통합 집계 | p.6 |

### 🔧 우리가 조정할 것

| 항목 | TRIP 방식 | 우리 조정 | TRIP 근거 |
|---|---|---|---|
| **속도 재계산** | 축분리 칼만 + SG 이중 평활 | 우리 `04_speed.py` 산출 평활 속도 사용. **SG 포함 여부 확인** → 없으면 06 전 SG 적용. 방위각도 04/05 정의가 북기준·2초평활과 일치하는지 검증 후 θv에 재사용 | p.4 |
| **PET 세그먼트 한정** | 전체 궤적 교차점에서 PET | 회전/횡단 등 **crossing course 세그먼트로 한정**(방위차≥30° + non-crossing 필터). 교차점 검출 방식(폴리라인 교차 vs 셀) 구체화 필요 | p.5 (§7 대조) |
| **이동류 OD 매칭** | junction#1만 OD 추출(#2는 ReID 부재로 곤란) | 송도 OD/turn 라벨링 **선행** → 각 이벤트에 자차·상대차 이동쌍 라벨 부착 → Fig.12형 이동쌍 랭킹 재현. 송도 입체가림 없으면 ReID 부담 경미 | p.6, p.8 |
| **차량치수 민감도** | 중심점-점 표현, 길이 미보정(TTC 과대추정 인정) | 송도 데이터에 **oriented bbox(길이·방위) 있으면** 중심거리에서 반길이 합 차감 보정 → TRIP 대비 **개선점**으로 서술. 없으면 점 표현 유지 + 한계 각주 | p.5, p.11(future work) |
| **임계값 정당화** | 정당화 회피(hot topic) | **`p=5%` 잡음바닥** 기반 단일 컷오프 정당화(우리 차별점). 단, 06 코드는 임계값을 **파라미터화**해 민감도 분석 가능하게 | p.3 |
| **다층 map-matching** | junction#2 층 구분 | 송도 단일 평면이면 **생략** | p.5 |
| **집계 해상도** | 5m 셀 × 30s, 99p 포락선+LOWESS | 우리 격자 설계 벤치마크로 참고, 지표별 표본 수 보고 결정 | p.10 |

---

## 7. 원문 대조가 더 필요한 항목 (재확인 목록)

깨끗한 PDF/원본으로 재대조 권장:

1. **⚠️ Eq.(1)(2)(3) 원본 표기** — 세 수식 모두 PDF 추출에서 분수/아래첨자 깨짐. 검증은 맥락으로 `confirmed`했으나 **분모 괄호·부호·아래첨자를 클린 소스로 최종 확인** 권장.
2. **θp·θc 벡터 구성 세부** — "거리벡터 vs 속도합/속도차" 각의 정확한 부호·기준(sub→surr 방향 등). 구현 시 90°/180° 경계에서 민감 → **원출처 Saunier et al.(2010) / Beauchamp et al.(2022) 대조**.
3. **동적 탐색공간 α 해석** — α가 **전체 부채꼴 각인지(±α/2) vs 반각인지** 원문에 명시 안 됨. 우리 implementation_note는 '전체 각(±α/2)'로 가정 → Fig.5a로 확인.
4. **PET 공통점 검출 방식** — "common spatial zone/point"를 **셀 기반인지, 폴리라인 기하 교차인지** 원문이 구현 수준으로 서술 안 함. 통과시각 보간 방법도 미상세.
5. **거리감소 조건 조작화** — `d(t)<d(t−1)` 단일 스텝인지, 구간 지속 감소인지 원문 불명확.
6. **Fig.11 볼록(convex) 거동** (confidence **medium**) — 검증용 정성 예시. 우리 검증 그림으로 재현해 정합성 점검(필수는 아님).
7. **관계분석 포락선 파라미터** — 99퍼센타일·LOWESS의 창/차수 등 구체값 미상세(Fig.13).
8. **임계값 인용 문장 페이지 경계** — "hot topic" 문장이 PAGE 2 말미~PAGE 3 첫머리에 걸침. 인용 시작은 p.3으로 귀속 확인됨(재확인 시 유의).