# PMB2 Guided SPU-BERT + Nav2 Runtime

이 문서는 HuNavSim에서 RViz 목표를 받아 PMB2를 움직이는 로봇 경로 실행 구조를
설명한다. 보행자 motion model과 로봇 path planner는 서로 독립적으로 선택한다.

## 실행 흐름

```text
RViz SetGoal (/goal_pose)
  -> spubert_nav2_bridge_node
     -> nav2 mode: bt_navigator가 /goal_pose를 직접 처리 (bridge는 상태만 관찰)
     -> monitor mode: Nav2 주행 + SPU-BERT 경로만 시각화
     -> spubert mode: guided SPU-BERT 12점 경로
        -> 안전성 검사
        -> Nav2 FollowPath
        -> 실패 시 NavigateToPose 폴백
  -> Nav2 controller
  -> /cmd_vel
  -> PMB2
```

SPU-BERT 노드는 `/cmd_vel`을 직접 publish하지 않는다. 실제 속도 명령과 로봇
kinematics 처리는 항상 Nav2 controller가 담당하므로 Nav2와 명령 충돌이 없다.

## 모드

- `nav2`: 기본값이다. 정적 장애물과 `/moai/human_obstacle_cloud`를 costmap에
  반영한 기존 Nav2가 목표까지 주행한다. 학습 체크포인트가 없어도 동작한다.
- `monitor`: Nav2가 주행하고 guided SPU-BERT의 12점 예측 경로를 RViz에
  표시한다. 모델을 실제 제어에 연결하기 전 권장하는 검증 모드다.
- `spubert`: 모델 경로를 Nav2 `FollowPath`로 실행한다. 모델 또는 경로가
  유효하지 않으면 기본 Nav2로 폴백한다.

## 기본 실행

```bash
cd hunavsim_containers/gazebo_classic

HUNAV_ROBOT_TYPE=pmb2 \
HUNAV_ROBOT_NAME=pmb2 \
HUNAV_AGENT_MOTION_MODEL=spubert \
HUNAV_NAVIGATION=True \
HUNAV_ROBOT_PATH_PLANNER=nav2 \
HUNAV_UPDATE_RATE=10.0 \
./run-hunav_gz_classic11_pmb2.bash
```

GPU가 없는 PC는 `HUNAV_DOCKER_USE_GPU=false`를 추가한다. 런처는 같은 이름의
오래된 `hunavsim_pmb2` 컨테이너를 먼저 제거하므로 재실행 시 name conflict가
발생하지 않는다.

GUI 없이 서버에서 실행할 때는 다음 두 값을 추가한다.

```bash
HUNAV_USE_GAZEBO_GUI=False \
HUNAV_USE_RVIZ=False \
```

## 모델 학습

guided 로봇 모델 설정은 다음 파일을 사용한다.

```text
moai_social_bert_refactored/configs/spubert/
  moai_social_nav_ext_scene_guided_fs.yaml
```

설정이 찾는 split 이름은 다음과 같다.

```text
pmb2_true_goal_gp_v1_clean_social_train.pkl
pmb2_true_goal_gp_v1_clean_social_val.pkl
pmb2_true_goal_gp_v1_clean_social_test.pkl
```

학습 예:

```bash
cd hunav_gz_classic_ws/src/moai_social_bert_refactored

GAZEBO_DATA_ROOT=/path/to/capstone/socially_aware_navigation/data/processed/gazebo/splits_episode \
./scripts/run_gazebo_train_docker.sh
```

기본 출력 체크포인트:

```text
output/spubert_moai_gazebo_guided_mgp_fs/model_best.pth
```

`output/ethucy/univ/spubert.pth`는 기존 보행자 motion model 체크포인트다. 새
guidance-conditioned 로봇 아키텍처와 parameter key가 다르므로 로봇 planner
체크포인트로 사용할 수 없다.

## 모델 주행 전 검증

먼저 `monitor`로 실행한다.

```bash
HUNAV_ROBOT_PATH_PLANNER=monitor \
HUNAV_ROBOT_SPUBERT_CHECKPOINT=/absolute/path/to/model_best.pth \
./run-hunav_gz_classic11_pmb2.bash
```

RViz에서 목표를 찍은 뒤 다음을 확인한다.

```bash
docker exec -it hunavsim_pmb2 bash -lc '
source /opt/ros/humble/setup.bash
source /home/pmb2_ws/install/setup.bash
source /home/hunav_gz_classic_ws/install/setup.bash
ros2 topic echo /moai/spubert_robot_planner_status
'
```

RViz 표시:

- `/moai/spubert_robot_path`: 선택된 12점 로봇 경로
- `/moai/spubert_robot_path_markers`: guidance 원, GP, final goal, 후보 goal,
  선택 경로와 reject 이유
- `/moai/human_obstacle_cloud`: Nav2 costmap에 들어가는 현재/예측 보행자 장애물

## 런타임 거절 원인 기록과 시각화

`scripts/run_simulation.sh <run_name>`으로 실행하면 각 예측의 후보 goal, TGP,
보행자 예측, map 충돌 step, 최대 step 거리, goal 진행량과 최소 보행자
안전거리가 다음 파일에 자동 저장된다.

```text
gazebo_classic/hunav_gz_classic_ws/moai_recordings/<run_name>_spubert_diagnostics.jsonl
```

실험이 끝난 뒤 호스트 PC에서 다음처럼 분석한다.

```bash
cd /home/kistmnl/social_nav/hunavsim_containers

python3 \
  gazebo_classic/hunav_gz_classic_ws/src/moai_social_bert_refactored/scripts/analyze_spubert_runtime_diagnostics.py \
  gazebo_classic/hunav_gz_classic_ws/moai_recordings/<run_name>_spubert_diagnostics.jsonl \
  --examples-per-reason 3
```

입력 파일 옆의 `<run_name>_spubert_diagnostics_analysis/`에 `summary.json`,
`reason_counts.csv`, `reason_counts.png`와 원인별 대표 경로 그림이 생성된다.

`guided_path_valid`가 반복해서 확인된 뒤 다음처럼 실제 모델 경로 실행을 켠다.

```bash
HUNAV_ROBOT_PATH_PLANNER=spubert \
HUNAV_ROBOT_SPUBERT_CHECKPOINT=/absolute/path/to/model_best.pth \
HUNAV_ROBOT_SPUBERT_FALLBACK_NAV2=True \
./run-hunav_gz_classic11_pmb2.bash
```

## 안전 검사와 폴백

모델 경로는 실행 전에 다음을 모두 통과해야 한다.

- 출력이 유한한 12개 좌표인지
- 로봇 footprint와 정적 occupancy map이 충돌하지 않는지
- 한 step 이동량이 설정된 최대 속도를 넘지 않는지
- final goal 방향으로 최소 progress가 있는지
- 보행자 예측 위치와 필요한 중심 간격을 유지하는지

모델 파일 누락, checkpoint 불일치, 맵 충돌, 보행자 clearance 실패,
`FollowPath` 서버 부재가 발생하면 상태 토픽에 이유를 publish하고 안전하게
기본 `NavigateToPose`로 전환한다.

## 맵 주의사항

Gazebo world와 RViz/Nav2 occupancy map은 별도 파일이다. 장애물을 수정할 때는
두 파일을 함께 갱신해야 한다.

```text
hunav_gazebo_wrapper/worlds/training_corridor.world
hunav_gazebo_wrapper/maps/training_corridor.pgm
hunav_gazebo_wrapper/maps/training_corridor.yaml
```

세 학습 맵은 `scripts/generate_training_maps.py`의 동일한 사각형 정의로
Gazebo collision과 occupancy grid를 함께 생성하므로 두 표현의 좌표가 일치한다.
