# Gazebo 주행 근본 원인 분석과 교수님 데모 기준

기준일: 2026-08-15

## 1. 결론

현재 Gazebo, 지도, 보행자, controller가 모두 고장 난 상태는 아니다.

- Route-choice에서는 기존 체크포인트가 Nav2 fallback 없이 장애물과 횡단
  보행자 2명을 지나 목표까지 도달했다.
- 더 큰 held-out Dual Route 장애물 앞에서는 MGP 후보 20개가 모두 장애물
  쪽에 생성되어 safety gate가 주행을 중단했다.
- 따라서 주원인은 **기존 straight-GP 학습 분포와 route-GP 런타임 입력의
  불일치, 장애물 우회 데이터 부족, collision loss 0.0**이다.
- safety gate가 경로를 거절하는 것은 오류가 아니라, 현재 모델의 unsafe
  출력을 실제 로봇에 보내지 않는 정상 동작이다.

## 2. 원인별 판정

| 구성 요소 | 확인 결과 | 판정 |
| --- | --- | --- |
| Gazebo world / Nav2 map | 같은 직사각형 정의에서 생성, 맵 관련 테스트 5/5 통과 | 좌표 불일치가 주원인 아님 |
| 보행자 | 기존 영상의 2명은 `cyclic_goals: false`라 목적지 도착 후 약 20초 정지 | 렉이 아니라 시나리오 수명주기 문제 |
| 혼잡/회전 | 같은 병목에 사람을 몰면 SFM 힘이 상쇄되고 계산량은 사람 수의 제곱에 가깝게 증가 | 사람 수보다 동선·시간 분산이 우선 |
| 상호작용 방향 | one-way coupling, 보행자는 로봇을 무시하고 로봇이 회피 | 데모의 인과관계는 명확함 |
| 저수준 controller | SPU-BERT TGP를 `FollowPath`로 받아 정상 추종 | 주원인 아님 |
| runtime safety gate | unsafe MGP/TGP를 거절하고 유효 경로에서 재출발 | 정상 동작 |
| 기존 체크포인트 | straight GP 5,952개, `col_weight: 0.0` | 핵심 학습 문제 |
| MGP | Dual Route 장애물 앞에서 20개 endpoint가 모두 unsafe | 핵심 실패 지점 |
| TGP 검사 | 기존에는 12개 점 사이 선분 충돌을 놓칠 수 있었음 | swept-footprint 검사로 수정 |

보행자 behavior가 모두 `Regular`인 것은 현재 단일 데모에는 사용할 수 있지만,
향후 야외 일반화 학습에는 부족하다. 정면 마주침, 횡단, 추월, 정지 후 재출발,
그룹, 가림 뒤 등장과 one-way/reciprocal 조건을 별도 metadata로 수집해야 한다.

## 3. 균형형 model-only 데모

시나리오:

```text
agents_professor_balanced_demo.yaml
robot: (-8.2, -1.4) -> (9.0, -1.4)
pedestrians: 순환 동선 4명(교차 2, offset oncoming 1, 배경 이동 1)
pedestrian max_vel: 0.70, 0.72, 0.62, 0.58 m/s
pedestrian start: 로봇의 navigation goal 수신 시점과 동기화
fallback: False
runtime seed: 21
```

사람 4명을 한 병목에 넣지 않고 서로 다른 차선과 도착 시각으로 분산했다.
따라서 화면에는 여러 사람이 계속 움직이지만, 로봇은 한 번에 1~2명만
판단하면 된다. 중앙의 큰 장애물은 그대로 두어 정적 우회 난도는 유지했다.
기존 안정화 설정의 `0.35~0.47 m/s`는 영상에서 지나치게 느렸으므로
`0.58~0.72 m/s`로 높였다. 속도만 높이면 로봇이 출발하기 전에 사람이
교차점을 통과하므로 `HUNAV_USE_NAVGOAL_TO_START=True`로 출발 시점을 맞췄다.

초기 고속 시험에서는 로봇이 올바르게 정지해 양보했지만, one-way coupling인
2번 보행자가 정지한 로봇을 그대로 관통해 최소 중심거리가 0.512 m까지
줄었다. 속도를 다시 낮추지 않고 해당 세로 통과선을 x=0.0에서 x=-1.3 m로
옮겼다. 이로써 2 m 이내의 가까운 상호작용은 유지하면서 물리적 겹침은
제거했다. 수정 경로는 보행자 반경을 포함한 지도 안전 검사도 통과했다.

2026-08-15 최종 속도 설정의 동일 조건 headless 반복 검증 결과:

| 지표 | Trial 1 | Trial 2 | Trial 3 |
| --- | ---: | ---: | ---: |
| 목표 도달 시간 | 27.5 s | 27.5 s | 27.5 s |
| 최종 남은 거리 | 0.497 m | 0.490 m | 0.541 m |
| valid prediction 비율 | 88.24% | 89.71% | 88.24% |
| Top-5가 1순위 실패를 복구 | 11회 | 17회 | 17회 |
| 사람 접근에 따른 hold | 7회 | 7회 | 8회 |
| Nav2 fallback | 0회 | 0회 | 0회 |
| 실제 최소 보행자 중심거리 | 1.460 m | 1.498 m | 1.497 m |
| 상호작용한 이동 보행자 | 3명 | 3명 | 3명 |
| 네 사람 최대 정지 시간 | 0.4 s | 0.4 s | 0.0 s |
| 제자리 회전 시간 | 0.0 s | 0.0 s | 0.0 s |
| 실제 직선 이탈량(장애물 우회) | 1.137 m | 1.170 m | 1.116 m |
| Gazebo RTF p10 | 0.904 | 0.867 | 0.905 |
| 전체 품질 gate | PASS | PASS | PASS |

세 trial 모두 네 사람이 전체 구간에서 실측 평균 `0.57~0.71 m/s`로 움직였고,
장시간 정지나 제자리 회전은 없었다. 로봇은 시작-목표 직선에서 1.1 m 이상
벗어나 중앙 장애물을 하단으로 우회했다. 사람과의 실제 최소 중심거리는
1.46 m 이상이었다. `fallback=False`이므로 완주는 Nav2의 대체 주행 결과가
아니라 SPU-BERT가 계속 공급한 경로의 결과다.

실행:

```bash
cd /home/junwoo/capstone/moai_social_bert_refactored
./gazebo_classic/run-professor-model-demo.bash professor_demo_video_01
```

`Automatic goal publisher completed 1 goals`가 나오면 촬영을 끝내고 `Ctrl+C`로
종료한다. 이 스크립트는 다음을 강제로 고정한다.

- `HUNAV_ROBOT_PATH_PLANNER=spubert`
- `HUNAV_ROBOT_SPUBERT_FALLBACK_NAV2=False`
- `HUNAV_ROBOT_SPUBERT_RUNTIME_SEED=21`
- `HUNAV_PEDESTRIANS_AVOID_ROBOT=False`
- `HUNAV_USE_NAVGOAL_TO_START=True`
- `HUNAV_ROBOT_SAVE_TRAINING_PKL=False`
- Gazebo GUI와 SPU-BERT debug RViz 동시 실행

따라서 완주하면 모델 경로로 주행한 것이고, 모델이 만든 demo trajectory가
expert 학습 데이터에 섞이지 않는다.

종료 후 로그까지 합격인지 확인한다.

```bash
./gazebo_classic/analyze-professor-model-demo.bash professor_demo_video_01
```

`Demo quality pass: True`여야 하며, 도착·model-only·4명 지속 이동·정지/회전·
이동 보행자 상호작용·장애물 우회·실제 안전거리·RTF를 모두 검사한다.

## 4. Dual Route가 실패한 이유

Dual Route 중앙 장애물은 x=2.0..7.0, y=-2.5..2.5인 5 m x 5 m 블록이다.
실패가 시작된 대표 시점은 다음과 같다.

```text
robot = (0.55, 1.49)
route GP = (7.81, 2.11)
footprint-safe MGP candidate = 0 / 20
```

당시 후보 20개의 y는 약 1.22..2.39 m에 몰렸다. 로봇 반경과 margin을
포함하면 블록 위를 지나기 위해 중심 y가 약 2.88 m보다 커야 하므로, 20개가
모두 거절되는 것이 맞다.

경로 GP 점 자체는 장애물 뒤의 전역 경로 위에 있었지만, 한 점만으로는 그
지점까지 가는 중간 굴곡을 충분히 표현하지 못한다. 기존 모델은 route GP와
이런 큰 코너를 함께 학습하지 않았기 때문에 GP 쪽으로 지름길을 생성했다.

## 5. 이미 해결한 런타임 문제

기존 footprint 검사는 TGP 12개 좌표만 검사했다. 좌표 사이가 길면 두 점은
안전해도 연결 선분이 장애물을 통과할 수 있었다.

현재는 로봇 현재 위치부터 TGP 12개 점까지의 모든 선분을 map resolution의
절반 간격으로 보간하고, 모든 sample에 footprint radius를 적용한다.

검증 결과:

```text
map / route GP / footprint / swept path unit tests: 21/21 passed
ROS build: hunav_gazebo_wrapper, moai_hunav_bridge passed
closed-loop model-only demo: reached, fallback 0
```

## 6. 근본 해결 순서

1. 교수님 영상은 위 Route-choice model-only 데모로 촬영한다.
2. Dual Route는 삭제하거나 학습에 넣지 않고 held-out stress test로 보존한다.
3. 새 데이터는 Nav2/assisted expert로 수집하고 route GP로만 후처리한다.
4. model-generated 실패 trajectory는 positive label로 사용하지 않는다.
5. Route-choice, bottleneck, chicane, intersection에서 먼저 150개 독립 episode를
   수집하고 품질 분포를 확인한다.
6. 최종적으로 최소 1,000개 성공 episode를 목표로 하며, 같은 episode의
   겹치는 window 수를 데이터 규모로 세지 않는다.
7. 동일 route-GP dataset으로 `col_weight=0.0, 0.1, 1.0`을 학습한다.
8. ADE/FDE/GDE뿐 아니라 endpoint-safe rate, swept collision rate, human
   clearance, model-only completion, fallback rate로 모델을 선택한다.
9. 마지막에만 Dual Route 및 학습에 없던 맵에서 3개 이상 seed로 반복 평가한다.

현재 단계에서 Dual Route가 완주하도록 safety margin을 줄이거나 collision
filter를 끄는 것은 해결이 아니다. 그런 변경은 실패를 숨기고 실제 로봇의
위험만 높인다.

## 7. 발표할 때의 정확한 표현

> Route-choice 시나리오에서는 기존 체크포인트도 경로 기반 GP와 Top-5 safety
> selection을 사용해 fallback 없이 장애물과 횡단 보행자를 통과했습니다.
> 반면 더 큰 held-out Dual Route에서는 MGP 후보가 모두 장애물 쪽에 생성되어
> safety gate가 정지시켰습니다. 따라서 런타임 구조의 가능성은 확인했지만,
> route-GP 데이터와 collision loss를 사용한 재학습이 일반화의 핵심 과제입니다.
