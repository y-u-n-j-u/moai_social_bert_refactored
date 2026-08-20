# 장애물 회피 + 보행자 상호작용 데이터 수집

## 목적

기존 데이터의 가장 큰 빈틈은 다음 두 조건을 동시에 포함한 샘플이 적다는 점이다.

1. 직선으로 갈 수 없어 경로 기반 guidance point가 필요한 정적 장애물
2. 그 우회 경로 안에서 양보, 교차, 반대 방향 통과가 필요한 보행자

따라서 보행자 수만 늘리지 않고, 한 에피소드에 의도된 상호작용을 1~2회 배치한다.
보행자는 목적지까지 한 번만 이동한다. 왕복 동선은 되돌아오는 시점에 우발적인
충돌을 만들 수 있으므로 joint 시나리오에서는 사용하지 않는다.

## 준비된 시나리오

| 시나리오 YAML | 맵 | 핵심 상황 |
|---|---|---|
| `agents_training_route_choice_joint_lower_crossing.yaml` | `training_route_choice` | 아래 통로 우회 + 수직 교차 |
| `agents_training_route_choice_joint_upper_crossing.yaml` | `training_route_choice` | 위 통로 우회 + 수직 교차 |
| `agents_training_route_choice_joint_lower_oncoming.yaml` | `training_route_choice` | 아래 통로 우회 + 반대 방향 통과 |
| `agents_training_route_choice_joint_upper_oncoming.yaml` | `training_route_choice` | 위 통로 우회 + 반대 방향 통과 |
| `agents_training_bottleneck_merge_joint_gate.yaml` | `training_bottleneck_merge` | 좁은 합류부 + 인접 반대 방향 + 교차 |
| `agents_training_outdoor_chicane_joint_crossing.yaml` | `training_outdoor_chicane` | S자 장애물 우회 + 세 교차 동선 |

`training_dual_route`는 최종 일반화 평가용이므로 수집 및 학습에 사용하지 않는다.

## 1. 파일만 먼저 확인

Gazebo를 띄우지 않고 선택한 시나리오, 맵, 시작점, 목표점을 확인한다.

```bash
cd /home/junwoo/capstone/moai_social_bert_refactored

HUNAV_PILOT_VALIDATE_ONLY=True \
HUNAV_JOINT_SCENARIO=agents_training_bottleneck_merge_joint_gate.yaml \
./gazebo_classic/run-joint-interaction-pilot.bash
```

## 2. 화면을 보며 한 번 확인

2D Goal Pose는 자동으로 발행되므로 RViz에서 직접 찍을 필요가 없다.

```bash
HUNAV_PILOT_NAME=bottleneck_joint_gui_smoke_20260810 \
HUNAV_JOINT_SCENARIO=agents_training_bottleneck_merge_joint_gate.yaml \
HUNAV_PILOT_SEEDS="501" \
HUNAV_PILOT_USE_GAZEBO_GUI=True \
HUNAV_PILOT_USE_RVIZ=True \
./gazebo_classic/run-joint-interaction-pilot.bash
```

노란 GP가 장애물 안이 아니라 Nav2 전역 경로의 안전한 통로 위에 있는지,
로봇과 보행자가 겹치지 않는지, 보행자가 첫 목적지에서 되돌아오지 않는지 확인한다.

## 3. 화면 없이 10-seed 파일럿 수집

```bash
HUNAV_PILOT_NAME=bottleneck_joint_pilot_v1_20260810 \
HUNAV_JOINT_SCENARIO=agents_training_bottleneck_merge_joint_gate.yaml \
HUNAV_PILOT_SEEDS="501 502 503 504 505 506 507 508 509 510" \
./gazebo_classic/run-joint-interaction-pilot.bash
```

다른 시나리오는 `HUNAV_JOINT_SCENARIO`와 `HUNAV_PILOT_NAME`만 바꾼다.
각 seed는 로봇 시작 위치를 조금씩 바꿔 보행자와 만나는 시점을 변화시킨다.

## 저장 결과

```text
gazebo_classic/hunav_gz_classic_ws/moai_recordings/<pilot_name>/
  pilot_summary.json
  seed_501/
    seed_501_raw.pkl
    qualification.json
    processed/
      seed_501_clean_all.pkl
      seed_501_clean_social.pkl
      seed_501_summary.json
      seed_501_quality.csv
```

학습 후보가 되려면 `pilot_summary.json`에서 다음을 만족해야 한다.

- `passed: true`
- 요청한 10개 seed가 모두 보고되고 통과
- 물리 충돌 거리 0.65 m 미만인 window가 0개
- route guidance 실패가 0개
- `clean_social` 샘플이 1개 이상

실패한 seed, 1-seed smoke 폴더, `pilot_summary.json`이 실패한 파일럿은 학습
PKL에 합치지 않는다.

## 구현 직후 실제 smoke 결과

| 시나리오 | 최소 보행자 거리 | raw | clean social | 결과 |
|---|---:|---:|---:|---|
| Route Choice lower crossing, seed 451 | 1.817 m | 9 | 6 | 통과 |
| Bottleneck merge, seed 461 | 1.804 m | 11 | 8 | 통과 |
| Outdoor chicane, seed 471 | 2.273 m | 14 | 5 | 통과 |

초기 Route Choice 왕복 동선은 0.388 m, 초기 병목 정면 동선은 0.035 m까지
접근해 품질 게이트에 실패했다. 각각 one-pass와 인접 반대 방향 동선으로 바꾼 뒤
통과했다. 이 결과는 보행자 수보다 상호작용의 위치, 방향, 타이밍 설계가 더
중요하다는 근거다.

## 권장 진행 순서

1. 여섯 시나리오를 각 3 seed로 먼저 실행해 실패 패턴을 찾는다.
2. 모두 안정적이면 시나리오별 10 seed 파일럿을 만든다.
3. 통과한 `clean_social.pkl`만 에피소드 단위로 train/val에 병합한다.
4. 기존 모델과 route-GP 모델을 같은 held-out `training_dual_route`에서 비교한다.
5. 그 다음 collision loss `0.0`, `0.1`, `1.0` 실험을 진행한다.

맵/경로/GP/보행자 동선 설계 그림은 다음 파일에 있다.

`figures/training_map_design/joint_interaction_scenarios.png`
