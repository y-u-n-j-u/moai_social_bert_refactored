# Route-choice Safe-Teacher 데이터 수집

## 목적

기존 학습 맵은 대부분 장애물 우회 방향을 선택하지 않아도 되는 구조였다. 반면
held-out `training_dual_route`는 중앙 장애물의 위/아래 중 한 경로를 선택해야 한다.
테스트 맵을 그대로 학습에 넣으면 일반화 평가가 오염되므로, 다른 형상의
`training_route_choice`를 학습 전용으로 추가했다.

```text
final goal
  -> Nav2 collision-free global path
  -> 현재 위치부터 path arc length 최대 8 m 지점
  -> route-aware guidance point
  -> robot/human trajectory와 local occupancy map 기록
  -> route GP 재계산 + recording 품질 게이트
  -> 통과 recording의 clean_social window만 학습 후보
```

`training_dual_route` recording은 계속 최종 closed-loop test 전용이며 train/val
PKL에 들어가면 안 된다.

## 학습 맵과 시나리오

`training_route_choice`는 22 m x 16 m이고 다음 요소를 가진다.

- 비대칭 중앙 블록: 중심 `(2.8, 0.8)`, 크기 `4.0 x 5.4 m`
- 위/아래 길이가 다른 두 우회 경로
- held-out Dual Route와 다른 장애물 위치, 크기, 전체 지도 크기
- start/goal 직선은 중앙 블록과 충돌하지만 Nav2 경로는 위/아래로 우회
- 낮은 쪽 보행자 lane `y=-4.0`, 높은 쪽 lane `y=5.1`
- one-way coupling: 보행자는 계획 경로를 유지하고 Nav2 teacher가 양보

안전 교사 시나리오는
`agents_training_route_choice_safe_teacher_low.yaml`이다. 보행자 lane을 로봇의
최단 우회 중심선과 분리해, 로봇이 정지하더라도 1.2 m 사회적 거리 기준을 지킬
공간을 남겼다.

## 실패에서 확인한 원인

### V1: 사람 lane과 장애물 margin이 모두 부족

첫 2-profile smoke는 0/2였다.

- 아래쪽: PMB2가 중앙 블록 코너를 물리적으로는 피했지만 데이터 기준
  `0.275 m robot radius + 0.10 m margin`을 만족하지 못함
- 위쪽: 보행자 lane이 로봇 우회 경로와 너무 가까워 최소 중심거리 0.86 m

따라서 품질 기준을 낮추지 않고 보행자 lane을 옮기고 route-choice 전용 Nav2
teacher 설정을 분리했다.

### V2: soft cost만으로는 일부 코너를 보장하지 못함

V2 2-profile smoke는 2/2였지만 10-profile pilot은 6/10이었다. 실패 4회는 모두
위쪽 우회이며 원인은 `target_map_collision` 하나였다. 사람 최소거리는 모두
1.56 m 이상이고 route guidance 실패는 없었다.

문제 좌표는 중앙 블록의 왼쪽 위 모서리 주변이었다. Nav2 local inflation을
0.55 m로 늘리고 CostCritic weight를 5.0으로 높여도, 이는 최적화 비용이므로
경로 길이를 줄이기 위해 margin 안쪽을 아주 짧게 통과할 수 있었다.

### V3: 수집용 virtual footprint를 hard constraint로 적용

실제 PMB2 반경은 0.275 m지만 route-choice teacher의 global/local costmap에는
`robot_radius: 0.40`을 사용한다. 데이터 요구 반경 0.375 m보다 큰 가상
footprint로 계획해, 0.10 m safety margin을 soft preference가 아닌 costmap
충돌 제약으로 만든다.

동일 seed 204, 동일 `y=+1.4` profile의 중앙 블록 코너 최소여유는 다음처럼
변했다.

| 설정 | 중앙 블록 코너 최소여유 | 데이터 기준 | 결과 |
| --- | ---: | ---: | --- |
| V2 soft margin | 0.335 m | 0.375 m | 실패 |
| V3 virtual footprint | 0.455 m | 0.375 m | 통과 |

시각화:
`figures/training_map_design/route_choice_teacher_v2_v3_clearance.png`

재생성:

```bash
python3 scripts/visualize_route_choice_teacher_clearance.py \
  --map-yaml gazebo_classic/hunav_gz_classic_ws/src/hunav_gazebo_wrapper/maps/training_route_choice.yaml \
  --v2 gazebo_classic/hunav_gz_classic_ws/moai_recordings/route_choice_safe_teacher_pilot_v2_20260810/seed_204/seed_204_raw.pkl \
  --v3 gazebo_classic/hunav_gz_classic_ws/moai_recordings/route_choice_safe_teacher_smoke_v3_profile3_20260810/seed_204/seed_204_raw.pkl \
  --out figures/training_map_design/route_choice_teacher_v2_v3_clearance.png
```

## 승인된 V3 결과

`route_choice_safe_teacher_v3_virtual_footprint_20260810` 결과:

| 항목 | 결과 |
| --- | ---: |
| 고유 start/goal profile | 10 |
| recording 품질 게이트 | **10/10 통과** |
| raw window | 83 |
| clean_all window | 83 |
| clean_social window | 38 |
| 전체 최소 사람 중심거리 | **1.561 m** |
| map/route/kinematic/timing 위반 | 0 |
| route guidance 실패 | 0 |

Aggregate 결과:
`gazebo_classic/hunav_gz_classic_ws/moai_recordings/route_choice_safe_teacher_v3_virtual_footprint_20260810/pilot_summary.json`

## 실행

기본 10-profile pilot:

```bash
cd /home/junwoo/capstone/moai_social_bert_refactored
gazebo_classic/run-route-choice-teacher-pilot.bash
```

특정 profile만 재현할 때는 offset을 지정한다. 다음은 V2에서 실패했던 profile 3이다.

```bash
HUNAV_PILOT_NAME=route_choice_profile3_recheck \
HUNAV_PILOT_SEEDS='204' \
HUNAV_PILOT_PROFILE_OFFSET=3 \
gazebo_classic/run-route-choice-teacher-pilot.bash
```

기존 결과 폴더는 덮어쓰지 않는다. 다른 이름을 사용해야 한다.

## 학습 데이터 승인 규칙

`prepare_gazebo_splits.py`의 기본 동작은 다음과 같다.

- 각 run의 `qualification.json`이 `passed: true`여야 함
- 부모 `pilot_summary.json`이 `passed: true`여야 함
- required/report/passed run이 각각 최소 10개여야 함
- 명시적으로 실패한 run은 legacy option에서도 항상 제외
- `training_dual_route`는 기본 held-out map으로 항상 제외
- `recording_id + episode_id` 단위로 split해 window leakage 방지

현재 전체 recordings를 입력하면 승인된 Intersection V3 36개와 Route-choice V3
38개만 남는다.

| split | sample | 독립 episode |
| --- | ---: | ---: |
| train | 50 | 14 |
| validation | 12 | 3 |
| internal test | 12 | 3 |
| 합계 | 74 | 20 |

이 internal test는 episode split 검사용이다. 최종 일반화 평가는 학습에 전혀
사용하지 않은 `training_dual_route`에서 별도로 수행한다.

## 해석과 다음 단계

이번 결과는 route-choice low-density 데이터 생성 설정이 승인됐다는 뜻이다.
새 SPU-BERT 모델의 성능이 개선됐다는 뜻은 아직 아니다. 38개 window만으로
재학습하면 route topology를 충분히 학습하기 어렵다.

다음 순서는 다음과 같다.

1. 같은 품질 게이트로 route-choice와 다른 training map에서 150개 독립 episode pilot 수집
2. 정면, 횡단, 추월, 정지/재출발과 medium-density safe-teacher를 각각 승인
3. 기존 5,952개 원본을 route GP로 재가공하고 품질 게이트 적용
4. episode-safe split 생성, Dual Route는 계속 held-out 유지
5. `col_weight=0.0, 0.1, 1.0` ablation 학습
6. Dual Route Low/Medium/High에서 success, collision, fallback, Top-K rescue 비교

실패한 V1/V2 recording은 삭제하지 않고 hard-negative 분석과 회귀 테스트에만
사용한다.
