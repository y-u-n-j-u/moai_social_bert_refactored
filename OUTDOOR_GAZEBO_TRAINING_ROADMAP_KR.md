# Gazebo 데이터에서 야외 실주행까지의 개발 로드맵

## 1. 결론

최종 시스템은 SPU-BERT가 단독으로 로봇을 제어하는 구조가 아니어야 한다.

```text
final goal
  -> Nav2 global route
  -> route를 따라 최대 8 m 앞 guidance point
  -> SPU-BERT MGP Top-K goal + TGP social path
  -> 정적 지도/footprint/보행자/kinematic safety gate
  -> Nav2 FollowPath controller
  -> 독립 collision monitor와 emergency stop
```

Nav2는 전역적으로 갈 수 있는 방향과 저수준 제어를 담당하고, SPU-BERT는 그 경로 주변에서 사람을 고려한 짧은 local path를 제안한다. 모델 출력은 항상 safety gate를 통과해야 하며, 모델이 실패하면 Nav2 fallback을 사용한다.

## 2. 현재까지 확인한 상태

| 항목 | 결과 | 판단 |
| --- | --- | --- |
| 기존 straight GP | 중앙 장애물 안에 들어갈 수 있었음 | 학습/런타임 입력 정책 오류 |
| Nav2 route GP | 장애물 없는 전역 경로를 따라 최대 8 m | 구현 완료 |
| MGP Top-5 + TGP 검사 | 기존 20-goal 실험에서 rank-1 실패 50회를 후순위 후보가 구제 | 단일 후보보다 유효 경로 증가 |
| 연속 3회 거절 후 fallback | fallback 19/20 -> 17/20 | 일시적 샘플 실패는 완화, 근본 해결은 아님 |
| Nav2/SPU-BERT 제어권 분리 | 1-goal smoke에서 41/41 valid, fallback 0 | 동시 action 선점 문제 해소 확인 |
| 학습용 route GP 후처리 | inflated occupancy route 기반 | 구현 및 단위 테스트 완료 |
| 기존 collision loss | 격자 충돌 개수만 세어 gradient가 없었음 | 학습에 실질적으로 작동하지 않음 |
| 새 collision loss | bilinear map sampling과 지도 밖 거리 페널티로 gradient 전달 | 구현 및 gradient 테스트 완료 |

1-goal smoke는 구조 검증일 뿐 일반화 성능 증명이 아니다. 동일 조건의 반복 seed, 보지 않은 맵, 고밀도 시나리오 검증이 추가로 필요하다.

### 2.1 2026-08-09 재부팅 후 체크포인트

코드와 결과 파일은 모두 디스크에 남아 있으며 다음 검증을 다시 통과했다.

| 검증 | 결과 |
| --- | --- |
| SPU-BERT 후보 선택/충돌 손실 단위 테스트 | 13/13 통과 |
| route guidance 런타임 단위 테스트 | 11/11 통과 |
| route GP 후처리/품질 게이트 단위 테스트 | 9/9 통과 |
| ROS 2 패키지 빌드 | `hunav_gazebo_wrapper`, `moai_hunav_bridge` 통과 |
| launch description 생성 | 통과 |

첫 번째 데이터 수집 전체 흐름 smoke는 Dual Route Low에서 Nav2 expert로 1 goal을 완료했다. 원본 4개 window 중 새 품질 게이트를 통과한 것은 2개였고 route 계산 실패와 swept-footprint 충돌은 0이었다. 다만 최소 보행자 거리가 약 4.1 m여서 `clean_social` 샘플은 0개였다. 즉, 파일 생성과 route-GP 후처리 연결은 확인했지만 사회적 상호작용 데이터 품질은 아직 확인하지 못했다.

Intersection Low에서는 수집 시작 동기화의 중요성이 확인됐다.

| Intersection Low 1-goal | 원본 window | `clean_social` | 전체 최소 중심거리 | 판단 |
| --- | ---: | ---: | ---: | --- |
| 보행자 선행 시작 | 6 | 0 | 0.539 m | Nav2 준비 전에 보행자가 움직여 전부 부적합 |
| 첫 goal과 보행자 시작 동기화 | 8 | 1 | 0.206 m | 1개는 1.223 m로 안전했지만 7개는 부적합 |

동기화된 유효 sample은 route GP, swept-footprint, 속도, 가속도, yaw-rate, timing 검사를 모두 통과했다. 그러나 같은 episode에서 보행자가 정지한 로봇을 통과하며 물리 반경 합보다 가까워지는 구간이 생겼다. 따라서 `HUNAV_USE_NAVGOAL_TO_START=True`는 수집 기본 조건으로 사용하되, 현재 Intersection Low를 그대로 positive 데이터 대량 수집에 사용하면 안 된다. 충돌 구간은 hard-negative/evaluation으로 보존하고, positive 수집에는 안전거리를 지키는 teacher와 시나리오 배치가 필요하다.

### 2.2 Intersection safe-teacher V3 검증

기존 맵과 `agents_training_intersection_low.yaml`은 baseline으로 보존했다. 별도
`agents_training_intersection_safe_teacher_low.yaml`에서 다음을 변경했다.

- 로봇을 교차로 바깥에서 출발시키고 보행자 1명이 수직 횡단
- one-way coupling으로 Nav2 teacher가 양보 책임을 가짐
- 사람 obstacle cloud 10 Hz, 2.4초 예측, 0.65 m 추가 margin
- teacher 전용 Nav2에서 짧은 후진 금지(`vx_min=0.0`)
- goal y를 좁은 corridor 중심선 기준 `-0.2..0.2 m`로 제한
- 실제 시작점-목표가 서로 다른 10개 encounter profile 사용

초기 탐색에서 y=+0.4 profile은 2개 중 하나가 map collision, 하나가 최소거리
1.167 m로 실패했다. 이 결과는 hard-negative/evaluation으로 보존하고 profile
범위를 줄였다. 최종 V3 결과는 다음과 같다.

| 항목 | 최종 결과 |
| --- | ---: |
| 고유 encounter profile | 10 |
| recording 품질 게이트 | **10/10 통과** |
| raw window | 63 |
| `clean_social` window | 36 |
| 전체 최소 사람 중심거리 | **1.332 m** |
| 물리/사회거리 위반 | 0 |
| map/route/속도/가속도/yaw/timing 위반 | 0 |
| 인프라 재시도 | 0 |

결과 파일은
`gazebo_classic/hunav_gz_classic_ws/moai_recordings/intersection_safe_teacher_v3_corridor_safe_20260810/pilot_summary.json`에 있다. 이 결과는 수집 파이프라인과 단일 횡단형 low-density teacher의 승인이지, 전체 맵과 보행자 행동 일반화가 완료됐다는 뜻은 아니다.

### 2.3 Route-choice safe-teacher V3 검증

held-out `training_dual_route`를 학습에 노출하지 않고 route topology를 학습시키기
위해 형상이 다른 `training_route_choice` 맵을 추가했다. start-goal 직선은 중앙
블록과 충돌하며, Nav2 route GP가 위/아래 우회 방향을 제공한다.

V1은 사람 lane과 장애물 margin 문제로 0/2였고, V2는 2-profile smoke는
통과했지만 10-profile에서 6/10만 통과했다. 실패 4회는 모두 중앙 블록의
왼쪽 위 코너에서 `target_map_collision`이 발생했다. 사람거리와 route GP는
정상이었으므로 원인은 모델이 아니라 Nav2 teacher의 soft clearance였다.

V3는 실제 반경 0.275 m 대신 수집 costmap에 0.40 m virtual footprint를 사용해
데이터 기준 0.375 m를 hard constraint로 만들었다. 동일 profile의 코너 최소
여유는 0.335 m에서 0.455 m로 증가했다.

| 항목 | Route-choice V3 |
| --- | ---: |
| 고유 encounter profile | 10 |
| recording 품질 게이트 | **10/10 통과** |
| raw / clean_all window | 83 / 83 |
| `clean_social` window | 38 |
| 전체 최소 사람 중심거리 | **1.561 m** |
| map/route/kinematic/timing 위반 | 0 |

상세 내용과 비교 그림은 `gazebo_classic/ROUTE_CHOICE_SAFE_TEACHER_KR.md` 및
`figures/training_map_design/route_choice_teacher_v2_v3_clearance.png`에 있다.

아직 완료되지 않은 항목은 medium/high-density safe-teacher 확장, 150-episode pilot 수집, 5,952개 원본 데이터 재가공, collision-loss ablation 학습, 보지 않은 맵의 반복 폐루프 평가, 실제 로봇 안전 계층 구현이다. 따라서 현재 상태는 **학습/수집 파이프라인 기반 구현 완료, 새 모델 학습과 야외 주행 검증 전**이다.

## 3. 현재 가장 큰 문제

1. 기존 5,952개 샘플은 straight GP 규칙으로 만들어져 새 런타임 규칙과 다르다.
2. 현재 이 저장소에는 5,952개를 만든 원본 PKL이 없어서 즉시 재생성할 수 없다.
3. 현재 training scenario의 보행자 behavior가 모두 `Regular`라서 야외 상호작용 다양성이 작다.
4. Sliding window stride 1은 거의 같은 창을 여러 번 학습시켜 샘플 수를 부풀린다.
5. ADE/FDE/GDE만 낮은 모델은 장애물 충돌률과 폐루프 주행 성공률이 나쁠 수 있다.
6. Gazebo ground truth만 학습하면 localization 오차, 센서 dropout, 지연, 노면 차이를 배우지 못한다.
7. 이전 SPU-BERT 평가는 one-way coupling, 최근 Nav2 expert pilot은 reciprocal coupling이었다. 두 데이터를 구분하지 않고 합치면 의미가 달라지므로 coupling policy를 manifest에 기록하고 두 조건을 모두 포함해야 한다.
8. Nav2가 목표에 도달해도 보행자가 로봇 반경 안으로 들어온 window가 존재한다. 목표 성공만으로 expert label을 승인하지 말고 동기화된 거리 검사를 반드시 통과시켜야 한다.

## 4. 데이터 수집 원칙

### 4.1 정답 trajectory를 만드는 주체

- 성공한 expert trajectory만 positive supervised label로 사용한다.
- Expert는 안정적인 Nav2 설정 또는 숙련된 assisted teleoperation을 사용한다.
- 현재 SPU-BERT가 만든 실패 경로를 다시 positive label로 학습시키지 않는다.
- timeout, 충돌, stuck episode는 별도 hard-negative/evaluation 파일로 보관한다.
- 목표에 도달한 episode의 window만 학습 파일에 commit한다.
- 보행자는 Nav2 준비 후 첫 goal과 동시에 시작하고 이 정책을 manifest에 기록한다.
- episode가 성공해도 안전거리 위반 window는 positive에서 제외해 hard-negative/evaluation으로 분리한다.

### 4.2 시나리오 축

| 축 | 반드시 포함할 값 |
| --- | --- |
| 지도 | 직선 보도, S자, 교차로, 병목/문, dual route, open plaza, blind corner, 경사/curb-ramp 근사 |
| 상호작용 | 정면 마주침, 횡단, 추월, 같은 방향, 정지 보행자, 그룹, 반대편에서 가려져 등장 |
| 밀도 | 1-2명, 3-5명, 6명 이상 stress |
| 방향 | 모든 시작/목표 방향, dual route 위/아래 균형 |
| 보행 속도 | 느림/보통/빠름, 정지와 재출발 포함 |
| 반응 | 로봇 무시, 회피, 주저함, 놀람, 접근 후 이탈 |
| 로봇 조건 | 초기 위치/방향, 목표 거리, 속도, 제어 지연 |

현재처럼 모든 agent를 `Regular`로 두지 말고 HuNav의 behavior와 SFM 계수를 시나리오 seed에 따라 섞는다. 물리적으로 불가능한 무작위화는 하지 않는다.

### 4.3 권장 수집 규모

먼저 150개 독립 episode의 pilot을 수집해 품질 분포와 rejection 원인을 확인한다. 기준을 고정한 뒤 최소 1,000개 독립 성공 episode를 목표로 한다. 중요한 단위는 겹치는 window 개수가 아니라 서로 다른 맵, 목표, pedestrian interaction, random seed를 가진 episode 수다.

기본 window stride는 4 frame으로 한다. `dt=0.4 s`이므로 새 학습 창은 1.6초 간격으로 시작한다. 희귀 interaction은 oversampling하되 같은 episode의 거의 같은 창을 복제하지 않는다.

## 5. 데이터 품질 게이트

현재 후처리 기본값은 런타임 제한에 맞춘다.

| 검사 | 기본 기준 |
| --- | --- |
| 시간 간격 | 목표 `0.4 s`, 최대 오차 `0.15 s` |
| 속도 | `<= 1.5 m/s` |
| 가속도 | `<= 2.5 m/s^2` |
| 경로 방향 변화율 | `<= 1.5 rad/s` |
| 정적 장애물 | robot radius `0.275 m` + margin `0.10 m`, 20개 pose 사이 swept path까지 충돌 0 |
| 보행자 중심 거리 | 20개 synchronized step 전체에서 `>= 1.2 m` |
| route GP | inflated map route 위, 최대 8 m, unknown/occupied 금지 |
| episode | goal 도달 episode만 positive dataset에 포함 |
| 상태 freshness | robot/human state age `<= 0.5 s` |

Pilot 결과를 보고 기준을 완화하거나 강화한다. 기준을 바꿀 때는 데이터 버전도 바꿔야 하며, rejection reason histogram을 함께 저장한다.

## 6. Split 정책

- 같은 episode의 window는 반드시 한 split에만 둔다.
- 같은 recording의 연속 episode도 가능하면 같은 split에 둔다.
- 최종 test에는 학습에 없던 map layout과 random seed를 둔다.
- 야외와 닮은 최종 2개 이상 맵은 학습에 절대 넣지 않고 closed-loop test 전용으로 보관한다.
- 보행자 수와 interaction category 비율을 train/validation에서 확인한다.

현재 `prepare_gazebo_splits.py`는 `recording_id + episode_id` leakage를 막고, `training_dual_route`를 기본 held-out map으로 제외한다. 또한 run qualification과 최소 10회 aggregate pilot 통과를 기본 요구해 failed pilot과 smoke data가 학습에 섞이지 않게 한다. 전체 recordings smoke에서 승인된 Intersection 36개와 Route-choice 38개만 남았고 50/12/12 sample, 14/3/3 episode로 분리됐다.

## 7. 학습 순서

1. 기존 ETH/UCY pretrained checkpoint를 초기값으로 사용한다.
2. 기존 5,952 raw PKL을 확보하고 route GP + 새 품질 게이트로 다시 후처리한다.
3. 데이터 prefix는 기존 straight-GP `v1`과 섞지 않고 `pmb2_route_gp_v2_clean_social`을 사용한다.
4. 동일 데이터와 seed 21로 `col_weight=0.0, 0.1, 1.0`을 비교한다.
5. 가장 좋은 weight를 선택한 뒤 seed 20, 21, 22를 학습한다.
6. 실패가 많은 obstacle corner, bottleneck, occlusion episode를 추가 수집해 hard-example fine-tuning한다.
7. 마지막에는 소량의 실제 야외 기록으로 낮은 learning rate adaptation을 수행한다.

준비된 1차 ablation config:

```text
configs/spubert/moai_social_nav_route_gp_continue_b21_col0.yaml
configs/spubert/moai_social_nav_route_gp_continue_b21_col01.yaml
configs/spubert/moai_social_nav_route_gp_continue_b21_col1.yaml
```

## 8. 모델 선택 지표

Offline:

- ADE, FDE, GDE
- MGP endpoint map-safe rate
- TGP 전체 swept-footprint collision rate
- minimum human clearance
- speed/acceleration/yaw-rate violation
- Top-1과 Top-K rescue rate

Gazebo closed-loop:

- goal success rate
- collision rate와 near-miss rate
- model-only completion rate와 Nav2 fallback rate
- path efficiency와 elapsed time
- stuck/recovery 횟수
- inference p50/p95 latency

ADE가 조금 낮다는 이유만으로 모델을 선택하지 않는다. 안전 조건을 통과한 모델 중에서 success와 social comfort가 좋은 모델을 선택한다.

## 9. 야외 투입 단계

1. Bag replay/shadow mode: 모델은 경로만 계산하고 실제 제어는 하지 않는다.
2. 울타리 친 빈 공간: 저속, 사람 없음, 물리 emergency stop 준비.
3. 한 명의 scripted pedestrian: 정면/횡단을 낮은 속도로 반복한다.
4. 통제된 다인 환경: safety operator와 원격 정지 장치 사용.
5. 실제 보도: 시간대와 밀도를 단계적으로 확장한다.

프로젝트 초기 제안 기준은 실제 충돌 0, emergency stop 정상 동작 100%, inference p95가 replan 주기보다 짧음, localization 불확실 시 즉시 감속/정지다. 최종 수치는 로봇 하드웨어와 운용 장소의 안전 검토 후 확정한다.

## 10. 바로 실행할 순서

1. 연구실 또는 백업에서 기존 5,952개를 만든 raw/processed PKL과 수집 manifest를 가져온다.
2. 새 route GP 후처리를 실행하고 `basic_failure_reasons`를 확인한다.
3. 완료: Intersection Low safe-teacher 10개 profile을 10/10 승인했다.
4. 완료: 별도 Route-choice 학습 맵과 virtual-footprint teacher를 10/10 승인했다.
5. 완료: failed/smoke pilot과 Dual Route를 자동 제외하는 quality + held-out split guard를 추가했다.
6. 다음: 정면/추월 및 medium-density teacher를 같은 방식으로 승인한 뒤 150-episode pilot을 수집한다.
7. old 5,952 route 재가공 데이터와 새 승인 pilot을 recording 단위로 합친다.
8. `col_weight=0.0, 0.1, 1.0` collision ablation 3개를 학습한다.
9. Dual Route Low/Medium/High와 보지 않은 맵에서 3-seed closed-loop 평가를 실행한다.

원본 PKL을 확보하기 전에는 새 모델 학습을 시작하지 않는다. 지금 저장소에 있는 checkpoint는 보존하고, 데이터와 config와 output 이름을 새 버전으로 분리한다.
