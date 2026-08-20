# 혼잡 완화형 Gazebo 데이터 수집 맵

## 목적

이 맵들은 보행자가 한 지점에 동시에 몰려 제자리 회전하거나 급격히 감속하고,
그 결과 로봇까지 장시간 정지하는 현상을 줄이면서 다음 학습 데이터를 추가로
수집하기 위해 만들었다.

- 장애물과 보행자를 함께 고려해야 하는 주행
- 넓은 공간의 다양한 상대 위치와 교차 상황
- 처음 보는 장애물 배치에서의 일반화 성능 평가

맵, Gazebo collision world, 보행자 scenario는
`scripts/generate_training_maps.py`의 동일한 정의에서 함께 생성된다. 따라서
occupancy map과 Gazebo 장애물 위치가 서로 달라지는 문제를 방지한다.

## 새 맵의 역할

| 맵 | 크기 | 장애물/통로 설계 | 권장 용도 |
|---|---:|---|---|
| `training_slalom` | 24 m x 18 m | 교대로 배치한 벽 3개, 우회 폭 약 3 m 이상, 분산된 보행자 횡단 지점 4개 | 학습 데이터 수집 |
| `training_open_plaza` | 24 m x 24 m | 1.4 m 섬형 장애물 4개, 넓은 자유 공간 | 학습 데이터 수집 |
| `training_route_choice` | 22 m x 16 m | 비대칭 중앙 블록, 상·하 우회 선택 | **학습 데이터 수집** |
| `training_dual_route` | 24 m x 18 m | 중앙에서 벗어난 5 m 블록, 상·하 경로 선택 | **미학습 테스트 맵** |

시각화: `figures/training_map_design/new_maps_overview.png`

각 맵에는 low/medium/high scenario가 있고 보행자 수는 각각 2/3/4명이다.
학습 수집은 처음부터 high만 사용하지 말고 low:medium:high를 대략
3:5:2 비율로 섞는 것을 권장한다.

## 혼잡과 급정지를 줄인 설정

생성된 모든 scenario에는 다음 값이 적용된다.

```yaml
max_vel: 1.10~1.25
radius: 0.33
goal_radius: 0.45
behavior:
  configuration: 1
  goal_force_factor: 2.0
  obstacle_force_factor: 5.0
  social_force_factor: 2.2
  other_force_factor: 6.0
```

가장 중요한 수정은 `configuration: 1`이다. HuNav의 `configuration: 0`은 YAML에
쓴 force factor 대신 더 강한 내장 기본값을 사용한다. 기존 생성기는 에이전트마다
0과 1을 번갈아 지정해서 일부 보행자만 장애물·사회적 반발력이 과도하게 커졌다.
현재는 모든 보행자가 명시된 완화값을 사용한다.

추가로 다음 원칙을 적용했다.

- 모든 보행자 시작점을 서로 떨어뜨림
- 모든 보행자 궤적을 장애물에서 `radius + 0.25 m` 이상 떨어뜨림
- Slalom에서는 보행자가 로봇 경로를 서로 다른 x 위치에서 횡단하도록 배치하고,
  여러 보행자의 교차 지점을 한곳에 집중시키지 않음
- 로봇과 보행자의 상호 회피가 서로를 계속 밀어내는 피드백을 막기 위해 기존의
  단방향 결합(`HUNAV_PEDESTRIANS_AVOID_ROBOT=False`) 유지
- Gazebo 장애물 force 비활성화와 30 Hz 보행자 갱신 설정 유지
- 현재 보행자 위치에 0.15 m 안전 여유를 추가하고, HuNav 보행자의 현재 속도를
  1.2초 앞까지 투영한 obstacle tube를 Nav2 local costmap에 전달해 횡단 상황에서
  로봇이 보행자를 더 일찍 감지하도록 함

맵 생성기는 PMB2 시작점, 자동 goal, 보행자 궤적의 장애물 여유를 생성 시 자동
검사하며 조건을 어기면 파일을 만들지 않고 오류를 낸다.

## 데이터 수집 방법

저장 이름은 맵·밀도·반복 번호가 드러나도록 정한다.

```bash
cd /home/kistmnl/social_nav/hunavsim_containers

HUNAV_AUTO_GOAL=True \
HUNAV_AUTO_GOAL_MODE=waypoint \
HUNAV_AUTO_GOAL_MAX_GOALS=20 \
HUNAV_AUTO_GOAL_MIN_EPISODE_DURATION=3 \
HUNAV_AUTO_GOAL_TIMEOUT=60 \
./scripts/run_simulation.sh slalom_medium_run_001
```

메뉴에서 저장 이름과 일치하는 scenario를 선택한다. 예를 들어 위 명령은
`agents_training_slalom_medium.yaml`을 선택한다. 자동 goal 20개가 끝나면
`Ctrl+C`를 눌러 시뮬레이션을 종료하고 메뉴에서 Exit를 선택한다.

수집 권장 순서:

1. `training_slalom` low/medium/high를 학습용으로 수집한다.
2. `training_open_plaza` low/medium/high를 학습용으로 수집한다.
3. `training_route_choice`는 `gazebo_classic/ROUTE_CHOICE_SAFE_TEACHER_KR.md`의
   승인된 virtual-footprint teacher로 수집한다.
4. recording과 10-run aggregate 품질 게이트를 통과한 파일만 기존 Gazebo
   데이터와 합쳐 B seed21 체크포인트에서 이어서 fine-tuning한다.
5. `training_dual_route`는 학습 데이터에 합치지 않고 최종 일반화 평가에만 쓴다.

## 구현 후 스모크 테스트 결과

GUI와 RViz를 끈 상태에서 자동 goal 두 개씩 실행했다.

| scenario | 보행자 | goal 1 | goal 2 | 결과 |
|---|---:|---:|---:|---|
| slalom high | 4 | 16.5 s | 27.5 s | 2/2 도달 |
| slalom crossing medium + 예측 obstacle | 3 | 17.0 s | 31.5 s | 2/2 도달, 최소 중심 거리 1.06 m |
| open plaza medium | 3 | 15.0 s | 17.5 s | 2/2 도달 |
| dual route medium | 3 | 16.0 s | 27.0 s | 2/2 도달 |

Dual-route 첫 목표에서는 MPPI가 한 번 경로 계산을 재시도했지만 costmap을 정리한
뒤 정상 복구했다. 모든 테스트에서 자동 목표 실패나 장시간 정지는 없었다.

Slalom 횡단 경로에서는 2 m 이내 사람-로봇 상호작용이 전체 거리 측정의 8.63%였고,
1 m 미만은 0%였다. 즉, 상호작용은 유지하면서 충돌성 근접은 제거했다.

이 결과는 짧은 실행 검증이다. 장시간 수집 중 품질은 에피소드별 성공률, 로봇
정지 시간, 보행자 평균 속도, 충돌/최소 거리로 다시 필터링해야 한다.
