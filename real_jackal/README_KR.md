# 실제 야외 Jackal SPU-BERT 배포

이 폴더는 동료의 센서 Docker와 현재 저장소의 최신
`Adaptive GP + collision loss 0.1 + social loss 0.05` 체크포인트를 연결한다.

## 데이터 흐름

```text
MID-360 + RealSense
  -> FAST-LIVO2 /aft_mapped_to_init
  -> YOLO + LiDAR fusion /ped_tracking
  -> pointcloud_to_laserscan /scan
  -> Nav2 ComputePathToPose global path (2초마다 갱신)
  -> Adaptive GP
  -> SPU-BERT MGP 20개
  -> footprint-safe 후보 중 Top-5 TGP
  -> 지도/동적 보행자/진행성 검사
  -> /spu_bert/predicted_path
  -> disarmed safety tracker
  -> 명시적으로 arm한 경우에만 Jackal cmd_vel
```

모델이 유효한 경로를 만들지 못하면 Nav2로 자동 주행하지 않고 정지한다. 초기 실차
시험에서 Nav2 fallback이 보행자를 무시하고 움직이는 것보다 이 동작이 안전하다.

## 1. 연구실 컴퓨터에 받기

```bash
cd /home/junwoo/capstone
git clone -b junwoo git@github.com:y-u-n-j-u/moai_social_bert_refactored.git
git clone git@github.com:y-u-n-j-u/spu_deploy_docker.git spu_deploy_docker_colleague

cd /home/junwoo/capstone/moai_social_bert_refactored
git lfs install
git lfs pull
```

체크포인트 확인:

```bash
sha256sum gazebo_classic/hunav_gz_classic_ws/src/moai_social_bert_refactored/output/spubert_route_gp_continue_b21_col01_social005/model_best.pth
```

기대값:

```text
03d88a96db9abe46c47d98837a57785f01c197a7338f0784510e88bec6a10a95
```

## 2. Docker 이미지 만들기

```bash
cd /home/junwoo/capstone/moai_social_bert_refactored
COLLEAGUE_REPO=/home/junwoo/capstone/spu_deploy_docker_colleague \
./real_jackal/docker/build_real_jackal_image.bash
```

첫 빌드는 동료 센서 base image까지 만들기 때문에 오래 걸린다. 이미
`spubert_deploy:latest`가 있으면 다음처럼 재사용한다.

이미지 빌드 마지막에는 CPU로 체크포인트를 실제 1회 추론한다. 이 단계가 실패하면
PyTorch/Transformers 버전 또는 모델 파일이 맞지 않는 것이므로 실차 시험을 진행하지
않는다.

```bash
SKIP_BASE_BUILD=1 ./real_jackal/docker/build_real_jackal_image.bash
```

## 3. 호스트 네트워크와 컨테이너

인터페이스 이름은 연구실 컴퓨터에서 `ip link`로 확인한 실제 값으로 바꾼다.

```bash
sudo nmcli device set <LIVOX_INTERFACE> managed no
sudo ip addr add 192.168.1.50/24 dev <LIVOX_INTERFACE>
sudo ip link set <LIVOX_INTERFACE> up
```

Jackal 본체가 domain 1을 쓰는지 먼저 확인한다. 확인되었다면:

```bash
cd /home/junwoo/capstone/moai_social_bert_refactored
ROS_DOMAIN_ID=1 ./real_jackal/docker/run_real_jackal_container.bash
docker exec -it moai_jackal_spubert bash
```

컨테이너 안에서 RViz를 표시하려면 컨테이너를 시작하기 전에 호스트에서 다음을
사용한다.

```bash
xhost +local:root
MOAI_ENABLE_GUI=1 ROS_DOMAIN_ID=1 \
  ./real_jackal/docker/run_real_jackal_container.bash
```

## 4. 센서와 보행자 추적 실행

컨테이너 터미널을 여러 개 열어 다음을 실행한다.

RealSense:

```bash
ros2 launch realsense2_camera rs_launch.py
```

MID-360은 RViz를 함께 띄우지 않는 launch를 사용한다.

```bash
ros2 launch livox_ros_driver2 msg_MID360_launch.py
```

Odometry와 보행자 perception:

```bash
ros2 launch mid360_bringup full_stack.launch.py \
  use_sim_time:=false use_fast_livo:=true launch_rviz:=false
```

`full_stack`이 static TF를 발행하므로 동일한 base-to-LiDAR TF를 별도 터미널에서
중복 실행하지 않는다.

LiDAR 필터와 2-D scan:

```bash
ros2 run mola_bringup filterpass.py --ros-args -p use_sim_time:=false
```

```bash
ros2 run pointcloud_to_laserscan pointcloud_to_laserscan_node \
  --ros-args \
  -r cloud_in:=/livox/lidar_filtered -r scan:=/scan \
  -p target_frame:=base_link -p transform_tolerance:=0.2 \
  -p min_height:=0.1 -p max_height:=1.5 \
  -p angle_min:=-3.14159 -p angle_max:=3.14159 \
  -p angle_increment:=0.0087 -p scan_time:=0.1 \
  -p range_min:=0.35 -p range_max:=30.0 \
  -p use_inf:=true -p inf_epsilon:=1.0 -p use_sim_time:=false
```

## 5. 야외 지도 준비

Adaptive GP는 장애물의 어느 쪽으로 우회할지 알려 줄 global path가 필요하다. 시험
구역은 먼저 teleop으로 한 번 매핑하고 map을 저장한다.

```bash
ros2 launch slam_toolbox online_async_launch.py use_sim_time:=false
```

맵을 충분히 만든 다음:

```bash
mkdir -p /root/jackal_maps
ros2 run nav2_map_server map_saver_cli -f /root/jackal_maps/outdoor_test
```

센서 장착과 odom TF에 맞춘 slam_toolbox 파라미터 조정은 실제 장비에서 필요할 수 있다.

## 6. Localization과 경로 planner

저장된 map으로 planner를 실행한다.

```bash
ros2 launch moai_jackal_spubert nav2_route_planner.launch.py \
  map:=/root/jackal_maps/outdoor_test.yaml
```

RViz에서 `2D Pose Estimate`로 초기 위치를 지정하고 다음을 확인한다.

```bash
ros2 run tf2_ros tf2_echo map odom
ros2 action info /compute_path_to_pose
```

## 7. Preflight와 monitor mode

아직 실제 cmd_vel에 연결하지 않는다.

```bash
/root/real_jackal_scripts/preflight_check.bash
```

```bash
ros2 launch moai_jackal_spubert real_jackal_spubert.launch.py \
  cmd_vel_topic:=/spu_bert/cmd_vel_dryrun \
  require_global_path:=true \
  launch_rviz:=true
```

RViz의 `Fixed Frame`이 `map`인지 확인한다. 먼저 `2D Pose Estimate`로 초기 위치를
지정하고, `2D Goal Pose`를 눌러 지도 위에 final goal과 방향을 지정한다. 이 도구가
`map` 좌표의 `PoseStamped`를 `/goal_pose`로 발행한다. 이후 다음을 확인한다.

- `/spu_bert/global_path_odom`: Nav2 global path
- `/spu_bert/debug_rolling_map`: 최신 LiDAR local map
- `/spu_bert/runtime_markers`: Adaptive GP, MGP 후보, 선택 TGP
- `/spu_bert/predicted_path`: 최종 12-step TGP, RViz에서 굵은 빨간 선
- `/spu_bert/runtime_status`: 추론 상태
- `/spu_bert/tracker_status`: 경로 추종 및 안전 정지 상태
- `/spu_bert/cmd_vel_dryrun`: 계산만 된 속도, 실제 로봇에는 전달되지 않음

`/scan`과 `/ped_tracking`의 `header.frame_id`는 모두 `base_link`여야 한다. 다른
좌표계가 들어오면 bridge와 tracker가 자동으로 정지한다.

최소 5분 monitor rosbag을 기록해 TF 점프, stale topic, 사람 좌표 방향, 경로 충돌이
없는지 확인한다.

```bash
ros2 bag record \
  /aft_mapped_to_init /scan /ped_tracking /goal_pose \
  /spu_bert/global_path_odom /spu_bert/predicted_path \
  /spu_bert/runtime_status /spu_bert/tracker_status /spu_bert/cmd_vel_dryrun
```

## 8. 저속 실차 제어

물리 E-stop을 잡은 담당자와 넓고 통제된 공간에서만 진행한다. launch를 실제 Jackal
topic으로 다시 실행해도 tracker는 기본적으로 disarm 상태다.

```bash
ros2 launch moai_jackal_spubert real_jackal_spubert.launch.py \
  cmd_vel_topic:=/j100_0519/cmd_vel \
  require_global_path:=true \
  launch_rviz:=true
```

먼저 실제 subscriber가 존재하는지 확인한다.

```bash
ros2 topic info /j100_0519/cmd_vel --verbose
```

bridge를 다시 실행했으므로 RViz의 `2D Goal Pose`로 final goal을 다시 지정한다.
`/spu_bert/runtime_status`가 `path_valid`이고 빨간 최종 TGP가 안전한 것을 확인한
뒤에만 arm한다.

모든 monitor 항목이 정상일 때만 arm한다.

```bash
ros2 service call /spu_bert/enable_motion std_srvs/srv/SetBool "{data: true}"
```

즉시 disarm:

```bash
ros2 service call /spu_bert/enable_motion std_srvs/srv/SetBool "{data: false}"
```

소프트웨어 emergency stop:

```bash
ros2 topic pub --once /spu_bert/emergency_stop std_msgs/msg/Bool "{data: true}"
```

시험 순서는 `바퀴를 띄운 상태 -> 0.25m/s 직선 -> 정적 장애물 -> 보행자 1명 ->
다중 보행자`로 올린다. 소프트웨어 정지는 물리 E-stop을 대체하지 않는다.

## 기존 동료 노드와의 관계

`ros2 run spu_data_extractor realtime_predict_node`와
`/root/moai_tracking_jackal/tracking_loop.py`는 비교 및 원본 보관용이다. 최신 모델 실차
시험에서는 이 둘 대신 `real_jackal_spubert_bridge`와 `safe_path_tracker`를 사용한다.

세부 감사 결과는 [DEPLOYMENT_AUDIT_20260826_KR.md](DEPLOYMENT_AUDIT_20260826_KR.md)에
정리되어 있다.
