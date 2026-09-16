# 실제 Jackal SPU-BERT 실행 가이드

기존 연구실 환경에 안정화 수정본을 적용할 때는
[Git clone부터 실차 적용까지](STABILITY_FIXES_KR.md)를 따른다.
`jackal-stability-20260916` 브랜치를 `$HOME/capstone_navi_stability`에 LFS 다운로드 없이
clone한 뒤 기존 센서/모델 이미지를 재사용해 ROS 패키지만 빌드한다.
기존 `/home/capstone_navi`를 덮어쓰지 않으며 아래 최초 설치 절차를 반복할 필요가 없다.
clone/pull만으로 이미 실행 중인 컨테이너의 Python 코드가 교체되지는 않는다.

이 문서는 `/home/capstone_navi`의 최신 guided SPU-BERT 모델과
MID-360/RealSense 센서 스택을 실제 Jackal에 연결하는 명령을 정리한다.

전체 흐름은 다음과 같다.

```text
MID-360 + RealSense
  -> FAST-LIVO2 /aft_mapped_to_init
  -> YOLO + LiDAR fusion /ped_tracking
  -> pointcloud_to_laserscan /scan
  -> Nav2 ComputePathToPose global path
  -> Adaptive GP + SPU-BERT
  -> /spu_bert/plan_context (goal + path; predicted_path는 RViz 표시용)
  -> disarmed safety tracker
  -> 명시적으로 arm한 경우에만 Jackal cmd_vel
```

모델이 유효한 경로를 만들지 못하거나 센서/TF가 stale이면 속도 0을 발행한다.
단, 일반적인 센서 stale·장애물 정지는 latch되지 않으므로 조건이 회복되면 arm
상태에서 자동 재개할 수 있다. 지속 정지는 반드시 disarm 또는 물리 E-stop을 사용한다.

## 0. 중요 안전 규칙

- 실제 주행은 물리 E-stop 담당자와 통제된 공간에서만 수행한다.
- 최초 시험은 바퀴를 띄운 상태에서 시작한다.
- 기존 `realtime_predict_node`, `tracking_loop.py`, teleop publisher를 최신 실차
  controller와 동시에 실행하지 않는다.
- `/scan`과 `/ped_tracking`의 `header.frame_id`는 모두 `base_link`여야 한다.
- 과거 사용되던 `msg_MID360_launch.py`는 사용하지 않는다. 이 launch는
  `CustomMsg`를 발행하지만 현재 perception/filter 스택은 `PointCloud2`를 요구한다.
- `full_stack.launch.py`의 identity odom fallback은 실차에서 끈다.
- 소프트웨어 정지는 물리 E-stop을 대체하지 않는다.
- 공유 노트북의 `~/.bashrc`에는 ROS 배포판, domain, RMW 또는 DDS profile을
  추가하지 않는다. 아래 전용 스크립트가 컨테이너 프로세스에만 적용한다.

## 1. 저장소와 모델 준비

이 절은 최초 설치용이다. 기존 `/home/capstone_navi`를 덮어쓰지 않는다.

메인 저장소:

```bash
cd /home
git clone -b jackal-stability-20260916 \
  https://github.com/y-u-n-j-u/moai_social_bert_refactored.git \
  capstone_navi
```

센서/캘리브레이션 저장소:

```bash
cd /home/capstone_navi
git clone \
  https://github.com/y-u-n-j-u/spu_deploy_docker.git \
  spu_deploy_docker
```

모델 받기:

```bash
sudo apt-get install -y git-lfs

cd /home/capstone_navi
git lfs install
git lfs pull --include="gazebo_classic/hunav_gz_classic_ws/src/moai_social_bert_refactored/output/spubert_route_gp_continue_b21_col01_social005/model_best.pth,gazebo_classic/hunav_gz_classic_ws/src/moai_social_bert_refactored/output/spubert_ethucy_all_scene_pretrain/pretrain_model_best.pth"
```

체크포인트 검증:

```bash
stat -c '%n %s bytes' \
  gazebo_classic/hunav_gz_classic_ws/src/moai_social_bert_refactored/output/spubert_route_gp_continue_b21_col01_social005/model_best.pth \
  gazebo_classic/hunav_gz_classic_ws/src/moai_social_bert_refactored/output/spubert_ethucy_all_scene_pretrain/pretrain_model_best.pth

sha256sum \
  gazebo_classic/hunav_gz_classic_ws/src/moai_social_bert_refactored/output/spubert_route_gp_continue_b21_col01_social005/model_best.pth
```

기대 크기와 SHA256:

```text
model_best.pth           91211704 bytes
pretrain_model_best.pth  44991593 bytes
03d88a96db9abe46c47d98837a57785f01c197a7338f0784510e88bec6a10a95
```

## 2. Docker 이미지 빌드

입력 검증:

```bash
cd /home/capstone_navi

COLLEAGUE_REPO=/home/capstone_navi/spu_deploy_docker \
VALIDATE_ONLY=1 \
./real_jackal/docker/build_real_jackal_image.bash
```

기존 `spubert_deploy:latest`가 있는지 확인한다.

```bash
docker image inspect spubert_deploy:latest
```

base image가 있으면 재사용한다.

```bash
COLLEAGUE_REPO=/home/capstone_navi/spu_deploy_docker \
SKIP_BASE_BUILD=1 \
./real_jackal/docker/build_real_jackal_image.bash
```

base image가 없을 때만 전체 빌드를 실행한다.

```bash
COLLEAGUE_REPO=/home/capstone_navi/spu_deploy_docker \
./real_jackal/docker/build_real_jackal_image.bash
```

최종 이미지와 GPU 확인:

```bash
docker image inspect moai-jackal-spubert:social005
docker run --rm --gpus all moai-jackal-spubert:social005 nvidia-smi
```

## 3. MID-360 네트워크 설정

인터페이스 이름은 `ip -br link`로 확인한 실제 값을 사용한다.
현재 MID-360 설정은 노트북 `192.168.1.50`, 센서 `192.168.1.130`을 전제로 한다.

```bash
ip -br link
ip -br addr

CAPSTONE_LIVOX_IF=enp3s0
sudo nmcli device set "$CAPSTONE_LIVOX_IF" managed no
sudo ip addr replace 192.168.1.50/24 dev "$CAPSTONE_LIVOX_IF"
sudo ip link set "$CAPSTONE_LIVOX_IF" up
ping -c 3 192.168.1.130
```

`CAPSTONE_LIVOX_IF=enp3s0`은 예시이므로 실제 인터페이스 이름으로 바꾼다.

실제 Jackal까지 연결하는 경우에는 위 명령 대신 프로젝트 전용 스크립트를 사용한다.
이 스크립트는 기존 주소, NetworkManager, route, gateway 및 DNS를 변경하거나
삭제하지 않고 누락된 Capstone 주소만 추가한다.

```bash
cd /home/capstone_navi

JACKAL_INTERFACE=enp131s0 \
./real_jackal/scripts/configure_jackal_network.bash up
```

다음 세 조건을 모두 통과해야 한다.

```text
Ethernet carrier is up
192.168.1.50, 192.168.131.50, 192.168.50.1 configured
Jackal DDS peer 192.168.50.2 reachable
```

Jackal 또는 Ethernet 케이블이 연결되지 않은 상태에서 carrier/ping 검사가 실패하는
것은 정상이다. 그 상태에서는 실제 주행 컨테이너를 시작하지 않는다.

## 4. 컨테이너 실행

Jackal의 실제 ROS domain을 확인한 뒤 `CAPSTONE_DOMAIN`에 지정한다.

```bash
cd /home/capstone_navi

CAPSTONE_DOMAIN=1
export CAPSTONE_CALIB=/home/capstone_navi/spu_deploy_docker/context/calib

test -s "$CAPSTONE_CALIB/extrinsic.txt"
test -s "$CAPSTONE_CALIB/intrinsic.txt"

xhost +local:root

MOAI_ENABLE_GUI=1 \
ROS_DOMAIN_ID="$CAPSTONE_DOMAIN" \
MAP_DIR=/home/moai/jackal_maps \
LOG_DIR=/home/moai/jackal_logs \
./real_jackal/docker/run_real_jackal_container.bash
```

위 명령은 Jackal 없이 센서만 점검하는 일반 실행이다. 실제 Jackal 주행에서는 기존
컨테이너를 종료한 뒤 정적 unicast DDS profile을 주입하는 전용 실행을 사용한다.

```bash
docker stop moai_jackal_spubert

cd /home/capstone_navi
MOAI_ENABLE_GUI=1 \
ROS_DOMAIN_ID=1 \
MAP_DIR=/home/moai/jackal_maps \
LOG_DIR=/home/moai/jackal_logs \
./real_jackal/docker/run_real_jackal_with_dds.bash
```

전용 실행은 이 저장소의 `fastdds_laptop_udp_discovery.xml`을 컨테이너에 read-only로 mount하고
다음 환경을 컨테이너에만 지정한다.

```text
ROS_DOMAIN_ID=1
ROS_LOCALHOST_ONLY=0
RMW_IMPLEMENTATION=rmw_fastrtps_cpp
FASTRTPS_DEFAULT_PROFILES_FILE=/root/jackal_runtime/fastdds_laptop.xml
FASTDDS_DEFAULT_PROFILES_FILE=/root/jackal_runtime/fastdds_laptop.xml
```

따라서 호스트 또는 다른 팀의 `~/.bashrc`에는 아무 설정도 추가하지 않는다.

`CAPSTONE_CALIB`을 지정한 실행은 calibration 디렉터리를 `/root/data/calib`에
read-only로 mount한다. 일반 실행에서 이 변수를 지정하지 않은 경우에만 다음 복사를
컨테이너를 새로 만들 때마다 반복한다.

```bash
docker exec moai_jackal_spubert mkdir -p /root/data/calib

docker cp "$CAPSTONE_CALIB/extrinsic.txt" \
  moai_jackal_spubert:/root/data/calib/extrinsic.txt

docker cp "$CAPSTONE_CALIB/intrinsic.txt" \
  moai_jackal_spubert:/root/data/calib/intrinsic.txt

docker exec moai_jackal_spubert ls -lh /root/data/calib
```

각 실행 터미널은 다음 명령으로 연다.

```bash
docker exec -it moai_jackal_spubert bash
```

실제 Jackal DDS 실행에서는 `.bashrc`를 전혀 읽지 않는 전용 shell을 연다.

```bash
cd /home/capstone_navi
./real_jackal/scripts/open_jackal_shell.bash
```

## 5. Jackal 없이 센서 스택 점검

Jackal 본체가 없어도 MID-360과 RealSense가 노트북에 전원·데이터로 연결되어 있으면
이 단계의 센서 토픽을 확인할 수 있다.

### 터미널 1: RealSense

```bash
ros2 launch realsense2_camera rs_launch.py
```

### 터미널 2: MID-360 PointCloud2

```bash
CAPSTONE_LIVOX_SHARE="$(ros2 pkg prefix --share livox_ros_driver2)"

ros2 run livox_ros_driver2 livox_ros_driver2_node --ros-args \
  -p xfer_format:=0 \
  -p multi_topic:=0 \
  -p data_src:=0 \
  -p publish_freq:=10.0 \
  -p output_data_type:=0 \
  -p frame_id:=livox_frame \
  -p "user_config_path:=${CAPSTONE_LIVOX_SHARE}/config/MID360_config.json" \
  -p cmdline_input_bd_code:=livox0000000001
```

반드시 타입을 확인한다.

```bash
ros2 topic type /livox/lidar
ros2 topic type /livox/imu
ros2 topic hz /livox/lidar
```

`/livox/lidar`는 반드시 `sensor_msgs/msg/PointCloud2`여야 한다.

### 터미널 3: FAST-LIVO와 보행자 perception

```bash
ros2 launch mid360_bringup full_stack.launch.py \
  use_sim_time:=false \
  use_fast_livo:=true \
  launch_rviz:=false \
  fast_livo_lidar_msg_type:=pointcloud2 \
  fast_livo_odom_fallback:=false \
  extrinsic_path:=/root/data/calib/extrinsic.txt
```

`full_stack`이 base-to-LiDAR static TF를 발행하므로 동일 TF를 별도로 발행하지 않는다.

### 터미널 4: LiDAR 필터

```bash
ros2 run mola_bringup filterpass.py --ros-args \
  -p use_sim_time:=false
```

### 터미널 5: 2-D LaserScan

```bash
ros2 run pointcloud_to_laserscan pointcloud_to_laserscan_node \
  --ros-args \
  -r cloud_in:=/livox/lidar_filtered \
  -r scan:=/scan \
  -p target_frame:=base_link \
  -p transform_tolerance:=0.2 \
  -p min_height:=0.1 \
  -p max_height:=1.5 \
  -p angle_min:=-3.14159 \
  -p angle_max:=3.14159 \
  -p angle_increment:=0.0087 \
  -p scan_time:=0.1 \
  -p range_min:=0.35 \
  -p range_max:=30.0 \
  -p use_inf:=true \
  -p inf_epsilon:=1.0 \
  -p use_sim_time:=false
```

### 센서 토픽과 frame 확인

```bash
ros2 topic type /livox/lidar
ros2 topic type /livox/imu
ros2 topic type /aft_mapped_to_init
ros2 topic type /scan
ros2 topic type /ped_tracking

ros2 topic echo /livox/lidar --once --field header.frame_id
ros2 topic echo /aft_mapped_to_init --once --field header.frame_id
ros2 topic echo /aft_mapped_to_init --once --field child_frame_id
ros2 topic echo /scan --once --field header.frame_id
ros2 topic echo /ped_tracking --once --field header.frame_id
```

정상 기대값:

```text
/livox/lidar         sensor_msgs/msg/PointCloud2, frame=livox_frame
/livox/imu           sensor_msgs/msg/Imu
/aft_mapped_to_init  nav_msgs/msg/Odometry, frame=odom, child=base_link
/scan                sensor_msgs/msg/LaserScan, frame=base_link
/ped_tracking        moai_nav_msgs/msg/Tracks, frame=base_link
```

TF 확인:

```bash
timeout 5s ros2 run tf2_ros tf2_echo odom base_link 2>&1 \
  | grep -m1 'Translation:'
```

## 6. 야외 지도 생성: 최초 1회

이 단계부터 Jackal과 검증된 joystick/teleop이 필요하다. 최신 SPU-BERT controller는
아직 실제 `cmd_vel`에 연결하지 않는다.

```bash
ros2 launch slam_toolbox online_async_launch.py use_sim_time:=false
```

시험장을 매핑한 후 다른 터미널에서 저장한다.

```bash
mkdir -p /root/jackal_maps
ros2 run nav2_map_server map_saver_cli \
  -f /root/jackal_maps/outdoor_test
```

저장 후 `slam_toolbox`를 `Ctrl-C`로 종료한다. SLAM과 AMCL의 `map->odom`을 동시에
실행하지 않는다.

## 7. Localization과 global planner

```bash
ros2 launch moai_jackal_spubert nav2_route_planner.launch.py \
  map:=/root/jackal_maps/outdoor_test.yaml
```

확인:

```bash
ros2 lifecycle get /map_server
ros2 lifecycle get /amcl
ros2 lifecycle get /planner_server
ros2 action info /compute_path_to_pose
```

세 lifecycle 노드는 모두 `active`여야 한다.

## 8. Dry-run monitor

아직 실제 Jackal cmd_vel에 연결하지 않는다.

```bash
ros2 launch moai_jackal_spubert real_jackal_spubert.launch.py \
  cmd_vel_topic:=/spu_bert/cmd_vel_dryrun \
  use_cuda:=true \
  require_global_path:=true \
  launch_rviz:=true
```

RViz에서 다음 순서로 지정한다.

1. `2D Pose Estimate`
2. `2D Goal Pose`

`map->odom`과 runtime 상태를 확인한다.

```bash
timeout 5s ros2 run tf2_ros tf2_echo map odom 2>&1 \
  | grep -m1 'Translation:'

ros2 topic echo /spu_bert/runtime_status
ros2 topic echo /spu_bert/tracker_status
```

`runtime_status`가 `path_valid...`이고 RViz 경로가 안전해야 한다.

launch는 항상 disarm으로 시작하므로 계산된 비영 속도를 확인하려면 dry-run만
명시적으로 arm해야 한다. 먼저 실제 구동 subscriber가 연결되지 않았는지 확인한다.

```bash
ros2 topic info /spu_bert/cmd_vel_dryrun --verbose
```

확인 후 dry-run arm:

```bash
ros2 service call /spu_bert/enable_motion \
  std_srvs/srv/SetBool "{data: true}"

ros2 topic echo /spu_bert/cmd_vel_dryrun
```

검증 후 disarm:

```bash
ros2 service call /spu_bert/enable_motion \
  std_srvs/srv/SetBool "{data: false}"
```

최소 5분 monitor rosbag은 컨테이너 삭제 후에도 남는 마운트 경로에 저장한다.

```bash
ros2 bag record -o /root/jackal_logs/monitor_$(date +%Y%m%d_%H%M%S) \
  /tf /tf_static /map /amcl_pose \
  /aft_mapped_to_init /scan /ped_tracking /goal_pose \
  /spu_bert/global_path_odom /spu_bert/predicted_path \
  /spu_bert/runtime_status /spu_bert/tracker_status \
  /spu_bert/cmd_vel_dryrun
```

## 9. 실제 Jackal 저속 주행

dry-run launch를 `Ctrl-C`로 완전히 종료한 후 실제 cmd_vel 토픽을 확인한다.

먼저 Jackal 컴퓨터에서 `clearpath-platform.service`와
`platform_velocity_controller`가 실제로 실행 중이어야 한다. 노트북의 전용
컨테이너 shell에서 다음 read-only preflight를 통과시킨다.

```bash
/root/jackal_runtime/scripts/verify_jackal_link.bash preflight
```

이 검사는 DDS peer와 다음 실제 Clearpath 명령 체인을 확인한다.

```text
/j100_0519/cmd_vel
  -> twist_mux
  -> /j100_0519/platform/cmd_vel_unstamped
  -> platform_velocity_controller
  -> j100_hardware_interface
  -> MCU
```

MCU motor feedback/command endpoint와 E-stop 토픽도 확인한다. `preflight` 중에는
E-stop이 눌린 상태여도 통과하며, 실제 launch 후 실행하는 `ready`에서는 해제 상태를
요구한다. 그 외 항목이 하나라도 실패하면 arm하지 않는다.

```bash
ros2 topic list -t | grep cmd_vel

CAPSTONE_CMD_TOPIC=/j100_0519/cmd_vel
ros2 topic type "$CAPSTONE_CMD_TOPIC"
ros2 topic info "$CAPSTONE_CMD_TOPIC" --verbose
```

`/j100_0519/cmd_vel`은 후보값이다. 실제 장비에서 다음을 모두 확인해야 한다.

- 타입이 `geometry_msgs/msg/Twist`
- 실제 Jackal `twist_mux` subscriber와 downstream platform controller/MCU 존재
- 경쟁 publisher 없음

실제 출력 launch도 항상 disarm으로 시작한다.

```bash
ros2 launch moai_jackal_spubert real_jackal_spubert.launch.py \
  cmd_vel_topic:="$CAPSTONE_CMD_TOPIC" \
  use_cuda:=true \
  require_global_path:=true \
  launch_rviz:=false
```

RViz에서 goal을 다시 지정하고 `path_valid` 및 경로 안전성을 확인한다. 물리 E-stop
담당자와 바퀴를 띄운 상태에서만 arm한다.

launch가 disarm 상태로 뜬 뒤 publisher/subscriber가 정확히 하나씩인지 다시 확인한다.

```bash
/root/jackal_runtime/scripts/verify_jackal_link.bash ready
```

```bash
ros2 service call /spu_bert/enable_motion \
  std_srvs/srv/SetBool "{data: true}"
```

즉시 disarm:

```bash
ros2 service call /spu_bert/enable_motion \
  std_srvs/srv/SetBool "{data: false}"
```

소프트웨어 emergency stop:

```bash
ros2 topic pub --once /spu_bert/emergency_stop \
  std_msgs/msg/Bool "{data: true}"
```

emergency stop은 latch된다. 다시 arm하려면 먼저 `SetBool false`로 latch를 reset한 뒤
`SetBool true`를 호출한다.

시험 순서는 다음과 같이 단계적으로 올린다.

```text
바퀴를 띄운 상태
  -> 0.25 m/s 직선
  -> 정적 장애물
  -> 보행자 1명
  -> 다중 보행자
```

## 10. 종료

```bash
ros2 service call /spu_bert/enable_motion \
  std_srvs/srv/SetBool "{data: false}"
```

각 ROS launch를 `Ctrl-C`로 종료한 뒤 호스트에서 컨테이너를 종료한다.

```bash
docker stop moai_jackal_spubert
xhost -local:root
```

지도는 `/home/moai/jackal_maps`, 로그와 rosbag은 `/home/moai/jackal_logs`에 남는다.

## 참고

- 세부 코드 감사: `DEPLOYMENT_AUDIT_20260826_KR.md`
- 실차 패키지: `real_jackal/ros2_ws/src/moai_jackal_spubert`
- 최신 모델 SHA256:
  `03d88a96db9abe46c47d98837a57785f01c197a7338f0784510e88bec6a10a95`
