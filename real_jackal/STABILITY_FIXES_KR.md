# 정지·저속 주행 안정화 코드의 연구실 적용

연구실의 기존 저장소는 `/home/capstone_navi`이다. 이 문서의 업데이트는 그 경로를
덮어쓰지 않고, GitHub의 `jackal-stability-20260916` 브랜치를 사용자 홈의 별도
디렉터리에 clone하여 기존 센서 이미지 위에
`moai_jackal_spubert` 패키지만 추가하는 방식이다. 모델 재학습, LFS 재다운로드,
MID-360/RealSense/perception base 이미지 전체 재빌드는 필요하지 않다.

clone은 소스를 받는 단계이며, 아래 overlay 이미지 빌드와 새 컨테이너 생성까지 해야
실행 코드에 반영된다. 기존 checkout의 미커밋 변경을 버리는 `git reset --hard`,
`git clean`, 강제 덮어쓰기는 하지 않는다. 새 checkout에는 LFS 포인터만 받아도 된다.
모델의 실제 가중치는 연구실의 기존 이미지에서 사용한다.

## 변경 범위와 검증의 의미

- 기본 `motion_guarded` 모드는 이동 방향을 신뢰할 수 있을 때 학습과 같은 마지막 이동
  선분의 heading을 사용하고, 정지·저속 잡음 구간에서는 odometry yaw를 사용한다.
- 추종점은 경로 선분을 따라 선택하고, 매번 안전 검사를 통과한 후보 사이의 연속성을
  비교한다. 이전 경로를 안전 검사 없이 계속 따르는 방식이 아니다.
- 목표 접근 감속과 도착 상태 공유, tracker JSON 진단을 추가한다.
- 추론은 별도 worker에서 수행하고, 완료 후 최신 지도·위치·보행자 정보로 다시 검사한다.
  경로 끝점을 이미 지난 경우 뒤로 돌아서지 않고 새 유효 경로를 기다린다.
- goal, path, goal ID를 한 `plan_context` 메시지로 전달한다. 별도 `/spu_bert/predicted_path`는
  기본 설정에서 RViz 표시용이다. 메시지 도착 순서 때문에 매 재계획마다 잠깐 멈추는 것을 피한다.
- 기존 AMCL/LIVO를 끄거나 TF를 고정하지 않는다. 실제 정지 중 TF/scan이 흔들리는
  원인을 해결했다는 뜻도 아니다. 아래 bag으로 그 문제를 별도로 확인한다.

이미지 빌드 중에는 소스 파일, 실제 Python import 위치/해시, 설치된 launch/config/RViz
파일의 해시를 비교한다. 이 검사는 **수정 코드가 배포됐다는 증거**다. 모델 성능이나
실차 안전을 검증하는 시험을 대신하지 않는다. Git commit이 같더라도 작업 중인 소스의
내용은 다를 수 있으므로 배포 기준은 `source_sha256`이다.

## 1. 새 브랜치를 clone하고 기존 환경을 기록한다

연구실 호스트 터미널에서 실행한다. `$HOME` 아래이므로 `/home`에 새 디렉터리를 만들
관리자 권한이 필요하지 않다. 목적지가 이미 있으면 `git clone`은 덮어쓰지 않고 실패한다.
그때는 새 디렉터리 이름을 선택하고 이후 경로도 맞춘다.

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --single-branch \
  --branch jackal-stability-20260916 \
  https://github.com/y-u-n-j-u/moai_social_bert_refactored.git \
  "$HOME/capstone_navi_stability" &&
cd "$HOME/capstone_navi_stability" &&
git branch --show-current &&
git rev-parse HEAD
```

브랜치 출력이 `jackal-stability-20260916`인지 확인하고 commit ID를 기록한다. 이 절차에서는
`git lfs pull`이나 별도 `spu_deploy_docker` clone을 하지 않는다. clone에 실패했다면
이후 단계를 계속하지 않는다.

받는 코드는 다음 위치에 있다. `/home/capstone_navi`의 원래 작업 내용과 분리된다.

```text
$HOME/capstone_navi_stability/
  real_jackal/
    docker/
    scripts/
    ros2_ws/src/moai_jackal_spubert/
    STABILITY_FIXES_KR.md
```

같은 호스트 터미널에서 기존 컨테이너를 기록한다. 이 스크립트는 컨테이너를 중단하거나
수정하지 않는다. `moai_jackal_spubert`가 실제 기존 컨테이너 이름과 다르면 바꾼다.

```bash
cd "$HOME/capstone_navi_stability"
SOURCE_CONTAINER=moai_jackal_spubert \
AUDIT_ROOT="$HOME/jackal_deployment_audits" \
bash real_jackal/scripts/audit_existing_deployment.bash
```

출력 디렉터리에 다음이 남는다.

- 컨테이너/이미지 ID와 설정, mount 목록, 컨테이너 writable layer 변경 목록
- 기존 source/installed/import 경로와 파일별 SHA256
- symlink가 가리키는 실제 패키지 파일 및 `/root/data/calib`의 tar 백업

이 audit의 import 경로는 새 shell에서 알려진 workspace 환경을 source하여 확인한
것이다. 이미 실행 중인 Python 프로세스의 메모리 내용을 증명하는 것은 아니다.
audit가 실패하면 출력이 부분적인 상태이므로 원인을 확인한 후 다시 기록한다.

기존 컨테이너가 이미 삭제됐으면 컨테이너 내부 수정 내용을 audit할 수 없다. 아래 목록에서
실제로 사용했던 이미지와 정확한 tag/ID를 확인해 `BASE_IMAGE`로 지정한다. 이미지 이름이
비슷하다는 이유로 기본 `social005`를 선택하지 않는다. 어느 이미지였는지 불명확하면
이미지/환경을 확인한 후 빌드를 진행한다.

```bash
docker ps -a --format '{{.Names}}  {{.Image}}  {{.Status}}'
docker image ls --no-trunc
# 확인된 이미지가 있는 경우에만: export BASE_IMAGE='확인한 이미지 tag 또는 ID'
```

기존 이미지에 없는 컨테이너 내부 수정은 `docker inspect`로 얻은 이미지 ID를
재사용하는 것만으로 보존되지 않는다. 어제 이미지/수정 내역이 불명확하면 다음과
같이 기존 컨테이너의 파일 시스템을 별도 이미지로 보관한 뒤 이를 base로 사용한다.
먼저 기존 SPU-BERT를 수동 disarm하고, 해당 ROS launch/센서 프로세스를 모두 종료해
컨테이너에는 대기 shell만 남긴다. `docker stop`은 기존 컨테이너가 `--rm`으로
생성됐다면 컨테이너를 삭제하므로 **아직 실행하지 않는다.**

```bash
# 기존 컨테이너 안: 운영자가 직접 수행한다.
ros2 service call /spu_bert/enable_motion std_srvs/srv/SetBool "{data: false}"
# 기존 ROS 실행 터미널에서 각각 Ctrl-C. 센서/추종 노드가 종료됐는지 확인한다.
```

```bash
# 호스트: 이름을 재사용하지 않는 새 snapshot tag를 지정한다.
export BASE_IMAGE="moai-jackal-spubert:lab-snapshot-$(date -u +%Y%m%d-%H%M%S)"
docker commit --pause=false moai_jackal_spubert "$BASE_IMAGE"
docker image inspect "$BASE_IMAGE" --format '{{.Id}}'
```

`docker commit`은 mount된 calibration/maps/logs를 이미지에 포함하지 않는다.
`mounts.json`의 기존 호스트 경로를 그대로 사용한다. calibration이 기존 컨테이너에만
있다면 컨테이너를 종료하기 전에 별도 디렉터리로 복사해 보존한다. 기존 파일 위에
덮어쓰지 말고 새 디렉터리를 사용한다. 모델/perception에 별도 bind mount를 사용했다면
그 내용도 snapshot에 포함되지 않으므로, 동일 입력을 재현한 base가 준비될 때까지
이 기본 절차를 진행하지 않는다.

이미지와 컨테이너에 차이가 없다는 것을 확인했다면 snapshot 대신 audit의
`base-image-id.txt` 값을 `BASE_IMAGE`로 지정할 수 있다. 이 문서의 도구들은 자동
snapshot, 기존 이미지 tag 덮어쓰기, 기존 컨테이너 중단을 수행하지 않는다.

## 2. ROS 패키지만 별도 이미지에 빌드한다

```bash
cd "$HOME/capstone_navi_stability"
export OUTPUT_IMAGE="moai-jackal-spubert:stability-$(date -u +%Y%m%d-%H%M%S)"
BASE_IMAGE="$BASE_IMAGE" OUTPUT_IMAGE="$OUTPUT_IMAGE" \
bash real_jackal/docker/build_stability_overlay.bash
```

빌더는 기존 base 이미지 ID를 새 보관용 tag로 고정하고,
`/root/moai_stability_ws`에 수정된 ROS 패키지만 빌드한다. 기존
`/root/robot_ws`와 모델 디렉터리는 base에서 이어받는다. 동일한 `OUTPUT_IMAGE`가
이미 있으면 덮어쓰지 않고 실패한다. 출력된 이미지명과 package SHA256을 기록한다.
빌드는 ROS 노드, 카메라/LiDAR, 경로 추종, arm을 실행하지 않는다. 기본 `motion_guarded`의
mm 단위 위치 잡음→yaw fallback과 0.2 m/s 이동→마지막 선분 heading을 검사하고,
CPU에서 실제 기존 checkpoint를 읽어 선택한 heading을 적용한 합성 입력 추론을 1회 수행한다.

개발 PC에서는 Docker/ROS 이미지 빌드와 실차 실행을 확인하지 못했다. 연구실에서
위 빌드의 import/manifest/모델 추론 검사가 성공하는지 확인해야 한다.
연구실 빌드에서 dependency/import 오류가 나면 먼저 base 이미지가
실제로 사용했던 sensor/model 환경인지 확인한다.

## 3. 새 컨테이너를 명시적으로 만든다

기존 컨테이너는 유지하되 기존 ROS 프로세스들이 종료됐는지 확인한다. 기존/새 컨테이너의
센서·localization·tracker 노드를 동시에 실행하지 않는다. 새 이름을 사용하면 기존
컨테이너를 삭제하지 않고도 적용 결과를 비교하고 되돌릴 수 있다.

아래 calibration/maps/logs는 **audit에서 확인한 기존 호스트 경로로 지정**한다.
저장소 경로가 `/home/capstone_navi`라는 사실만으로 기존 지도/로그 경로도 그 아래라고
가정하지 않는다. 예를 들어 기존 지도 경로가 `/home/moai/jackal_maps`이면 그대로 쓴다.

아래 설정 예시는 export만 하며 네트워크·컨테이너·로봇을 조작하지 않는다. `BASE_IMAGE`와
`OUTPUT_IMAGE`도 변경하지 않으므로 앞 단계와 **같은 호스트 터미널**에서 이어서 실행한다.
다른 호스트 터미널로 옮겼다면 앞에서 빌드한 정확한 tag를 `OUTPUT_IMAGE`에 다시 지정한다.

```bash
cd "$HOME/capstone_navi_stability"
source real_jackal/config/stability_host.env.example

export CONTAINER_NAME=moai_jackal_spubert_stability
# audit의 mount 경로가 예시와 다르면 아래 세 값을 실제 경로로 바꾼다.
export CAPSTONE_CALIB=/home/capstone_navi/spu_deploy_docker/context/calib
export MAP_DIR="$HOME/jackal_maps"
export LOG_DIR="$HOME/jackal_logs"
test -s "$CAPSTONE_CALIB/extrinsic.txt"
test -s "$CAPSTONE_CALIB/intrinsic.txt"

xhost +local:root
IMAGE="$OUTPUT_IMAGE" \
bash real_jackal/docker/run_stability_container.bash
```

새 컨테이너는 `sleep infinity`로 대기한다. 실행 중인 컨테이너를 덮어쓰거나 자동으로
ROS를 launch하지 않는다. 이름을 기존 `moai_jackal_spubert`로 지정하는 것도 가능하지만,
기존 컨테이너가 있으면 명시적으로 실패한다. 새 이름을 권장한다.

일반 `run_real_jackal_container.bash`와 달리 stability 실행에서는 scripts와 Nav2 YAML을
호스트에서 덮어 mount하지 않는다. 검증한 이미지 안 파일을 사용한다. DDS XML과
기존 calibration/maps/logs는 별도로 mount한다. 모델·Python·파라미터 수정 후에는
새 image tag로 다시 빌드해야 하며, clone/pull만으로 실행 코드가 바뀌지는 않는다.

각 새 터미널은 다음으로 연다.

```bash
cd "$HOME/capstone_navi_stability"
CONTAINER_NAME=moai_jackal_spubert_stability \
bash real_jackal/scripts/open_jackal_shell.bash
```

여기서 `CONTAINER_NAME`은 새 컨테이너를 만들 때 사용한 값과 같아야 한다. 예전 실행
명령의 `run_real_jackal_with_dds.bash`를 `IMAGE` 지정 없이 실행하면 기본 `social005`
이미지로 돌아가므로 이번 업데이트에서는 위 `run_stability_container.bash`를 사용한다.
호스트 스크립트는 `bash ...`로 호출하므로 clone 환경의 실행권한 차이로 막히지 않는다.

컨테이너 안에서 먼저 확인한다.

```bash
ros2 pkg prefix moai_jackal_spubert
# /root/moai_stability_ws/install/moai_jackal_spubert 이어야 한다.
/usr/bin/python3 /root/jackal_runtime/scripts/deployment_manifest.py verify
```

## 4. 기존 센서 순서를 유지하고 먼저 dry-run한다

기존 실행 문서의 터미널 1~6 순서(RealSense → MID-360 PointCloud2 → FAST-LIVO/perception
→ filter → LaserScan → 저장 지도/AMCL/Nav2)를 유지한다. `fast_livo_odom_fallback:=false`,
캘리브레이션 파일, 기존 지도 파일을 그대로 사용한다. 새 DDS 설정은 기존 장비 주소
`192.168.50.1`/peer `.2`, 기본 domain `1`을 전제로 하므로 실험실 실제값과 확인한다.

아래 각 명령은 위의 `open_jackal_shell.bash`로 연 **서로 다른 터미널**에서 실행한다.
첫 다섯 터미널의 센서·인식이 동작한 뒤 지도와 모델을 시작한다.

```bash
# 터미널 1: 카메라
ros2 launch realsense2_camera rs_launch.py
```

```bash
# 터미널 2: 라이다 (PointCloud2 출력)
CAPSTONE_LIVOX_SHARE="$(ros2 pkg prefix --share livox_ros_driver2)"
ros2 run livox_ros_driver2 livox_ros_driver2_node --ros-args \
  -p xfer_format:=0 -p multi_topic:=0 -p data_src:=0 \
  -p publish_freq:=10.0 -p output_data_type:=0 -p frame_id:=livox_frame \
  -p "user_config_path:=${CAPSTONE_LIVOX_SHARE}/config/MID360_config.json" \
  -p cmdline_input_bd_code:=livox0000000001
```

```bash
# 터미널 3: 위치 추정과 보행자 인식
ros2 launch mid360_bringup full_stack.launch.py \
  use_sim_time:=false use_fast_livo:=true launch_rviz:=false \
  fast_livo_lidar_msg_type:=pointcloud2 fast_livo_odom_fallback:=false \
  extrinsic_path:=/root/data/calib/extrinsic.txt
```

```bash
# 터미널 4: 라이다 필터
ros2 run mola_bringup filterpass.py --ros-args -p use_sim_time:=false
```

```bash
# 터미널 5: 2D 스캔
ros2 run pointcloud_to_laserscan pointcloud_to_laserscan_node --ros-args \
  -r cloud_in:=/livox/lidar_filtered -r scan:=/scan \
  -p target_frame:=base_link -p transform_tolerance:=0.2 \
  -p min_height:=0.1 -p max_height:=1.5 \
  -p angle_min:=-3.14159 -p angle_max:=3.14159 -p angle_increment:=0.0087 \
  -p scan_time:=0.1 -p range_min:=0.35 -p range_max:=30.0 \
  -p use_inf:=true -p inf_epsilon:=1.0 -p use_sim_time:=false
```

```bash
# 터미널 6: 저장 지도, AMCL, 전역 경로 생성기
# world_map.yaml은 실제 사용하는 저장 지도 파일명으로 맞춘다.
ros2 launch moai_jackal_spubert nav2_route_planner.launch.py \
  map:=/root/jackal_maps/world_map.yaml use_sim_time:=false
```

SPU-BERT는 다음으로 시작한다. 시작 상태는 disarm이고 아래 명령은 arm하지 않는다.

```bash
ros2 launch moai_jackal_spubert real_jackal_spubert.launch.py \
  cmd_vel_topic:=/spu_bert/cmd_vel_dryrun \
  use_cuda:=true require_global_path:=true launch_rviz:=false
```

별도 RViz를 사용할 경우 overlay가 선택한 설정을 연다.

```bash
rviz2 -d "$(ros2 pkg prefix --share moai_jackal_spubert)/rviz/real_jackal_spubert.rviz"
```

먼저 초기 위치와 goal을 지정한 후, 별도 shell에서 실제 파라미터와 진단을 기록한다.

```bash
CMD_TOPIC=/spu_bert/cmd_vel_dryrun \
bash /root/jackal_runtime/scripts/capture_stability_state.bash
```

로그 디렉터리의 `bridge-parameters.yaml`/`tracker-parameters.yaml`에 다음을 확인한다.

| 항목 | 이번 기본값 |
| --- | --- |
| `model_heading_mode` | `motion_guarded` |
| `inference_max_age_sec` | `1.20` (추론 결과 최대 나이; 기존 센서 timeout과 별개) |
| `candidate_selection_mode` | `continuous` |
| `goal_slow_distance` | `1.5` |
| `goal_approach_minimum_speed` | `0.05` |
| tracker `diagnostics_topic` | `/spu_bert/tracker_diagnostics` |
| tracker `require_plan_context` | `true` |
| bridge/tracker `check_sensor_header_stamps` | `true` |
| tracker `cmd_vel_topic` | 처음에는 `/spu_bert/cmd_vel_dryrun` |

`capture_stability_state`는 각 조회를 30초로 제한한다. `[INCOMPLETE]`는 최신 메시지가
없거나 discovery/노드 상태 때문에 기록을 못 받은 경우이므로 성공으로 간주하지 않는다.
단순히 오래된 터미널 출력만 보고 현재 상태를 판정하지 않는다.
`goal-completion.txt`는 완료 이벤트가 아직 없으면 비어 있는 것이 정상이며 필수 검사
실패로 세지 않는다. `plan-context.txt`는 goal 지정 후 확인한다.

센서는 수신시각뿐 아니라 원래 `header.stamp`도 검사한다. `odom_stamp_stale`,
`scan_stamp_stale`, `tracks_stamp_stale`, `*_stamp_in_future`가 발생하면 원본 데이터 지연과
ROS 시계를 확인한다. 0/누락 stamp도 거부한다. bridge의 `inputs_stale_after_inference` 또는
`inputs_stale_after_validation`은 추론·검사가 끝날 때 입력이 오래됐다는 뜻이다. 기존
센서 timeout을 임의로 늘려 해결하지 않고, 진단의 `inference_duration_sec`, 추론에 사용한
입력 시각과 최신 센서 age를 같이 본다.

추론 중에도 센서 콜백은 계속 동작한다. 완료 후 최신 정보로 안전 검사를 다시 하며,
`inference_result_expired`는 추론 결과가 1.20초를 넘었다는 뜻이다. 그 나이는 ROS 시각과
단조 시계 중 더 오래된 쪽으로 계산한다. 결과 재검증·발행 후에는 기존 tracker의
경로 수명 1.20초가 적용된다. `model_map_generation`/입력 시각과 `validation` 안의
지도 세대·센서 시각을 구별해서 본다. 아직 수집하지 않은 모델 결과는
`model_output_collected=false`로 기록되며 실제 사용 heading이 없을 수 있다.

0.50m 지도 반경과 0.75m 전방 정지의 차이, 확인된 계산 문제와 영상 분석의 한계는
[STOP_YAW_CAUSE_REVIEW_KR.md](STOP_YAW_CAUSE_REVIEW_KR.md)에 정리했다.

기본 모드는 신뢰할 수 있는 이동 구간에서 학습과 같은 heading을 유지한다. 저속 구간의
yaw fallback은 이동 방향 정규화와 다를 수 있다. 좌표변환을 검사했더라도 기존 checkpoint의
실제 경로 품질 개선은 실험으로 확인해야 한다.

## 5. 비교 기록과 실제 주행 전환

먼저 물리 로봇을 정지시킨 상태를 포함해 기록한다. 실제 드라이브 토픽으로 바꾸기 전에는
dry-run 토픽의 endpoint를 확인하고, 기존 README의 dry-run arm/disarm 절차를 운영자가
직접 수행한다. 아래 bag은 상태/좌표계 문제를 함께 확인하기 위한 것이며 주행을 시작하지 않는다.

```bash
ros2 bag record -o /root/jackal_logs/stability_$(date -u +%Y%m%d_%H%M%S) \
  /tf /tf_static /map /amcl_pose /aft_mapped_to_init \
  /livox/lidar /livox/imu /scan /ped_tracking /goal_pose \
  /spu_bert/global_path_odom /spu_bert/final_goal_odom \
  /spu_bert/predicted_path /spu_bert/debug_rolling_map \
  /spu_bert/runtime_status /spu_bert/tracker_status /spu_bert/tracker_diagnostics \
  /spu_bert/plan_context /spu_bert/goal_completion \
  /spu_bert/cmd_vel_dryrun /j100_0519/cmd_vel
```

합격 판단은 다음을 함께 본다.

1. 정지·저속에서 모델 heading과 odometry yaw가 일치하고 작은 위치 차분만으로 급회전하지 않는가.
2. 경로가 조금 변할 때 추종점·각속도가 반복적으로 좌우로 튀지 않는가. 안전하지 않은 후보는 계속 거부되는가.
3. 목표 접근 시 감속하고, 도착 disarm 후 새 goal을 주기 전까지 완료 상태가 유지되는가.
4. 실제 정지 중 `odom→base_link`, `map→odom`, scan/저장 지도의 상대 위치가 계속 변하지 않는가.
   이 항목이 실패하면 모델 수정으로 localization 문제가 해결됐다고 판단하지 않는다.
5. stale 센서, 경로 없음, 장애물 조건에서 속도 0과 해당 진단 이유가 일치하는가.

실제 주행은 기존 dry-run launch를 완전히 종료한 뒤, 운영자가 기존 README의
`verify_jackal_link.bash preflight`를 확인하고 아래처럼 **출력 토픽만 명시적으로** 바꾼다.

```bash
ros2 launch moai_jackal_spubert real_jackal_spubert.launch.py \
  cmd_vel_topic:=/j100_0519/cmd_vel \
  use_cuda:=true require_global_path:=true launch_rviz:=false
```

이때도 시작은 disarm이다. 기존의 E-stop 담당자/바퀴를 띄운 초기 점검과 `ready` 확인,
운영자의 수동 arm 절차를 유지한다. 본 문서의 추가 스크립트는 자동 arm, 자동 goal,
자동 속도 발행을 하지 않는다.

RViz에서 실제 위치·방향을 초기화하고 새 goal을 지정한다. 유효 경로와 출력 토픽을
확인한 뒤 운영자가 별도 shell에서 다음을 실행하면 실제 주행을 허용한다.

```bash
ros2 service call /spu_bert/enable_motion std_srvs/srv/SetBool "{data: true}"
```

도착하면 자동 disarm된다. 수동 정지는 다음과 같다. 같은 완료 goal로 다시 arm할 수
없으며 다음 주행에는 새 goal을 지정한 뒤 다시 arm한다.

```bash
ros2 service call /spu_bert/enable_motion std_srvs/srv/SetBool "{data: false}"
```

주행 후 컨테이너에서 JSONL 통계를 만든다. 같은 로그 파일에 여러 주행이 이어져 있으면
한 주행 범위로 분리해서 비교한다.

```bash
python3 /root/jackal_runtime/scripts/summarize_stability_logs.py \
  --bridge /root/jackal_logs/spubert_real_jackal_diagnostics.jsonl \
  --tracker /root/jackal_logs/tracker_stability.jsonl \
  --output /root/jackal_logs/stability_report.json
```

집계는 후보 탈락, 실제 모델 theta, 요청/발행 각속도 부호 반전, 소프트웨어 정지 상태의
표본 간 시간이다. 실제 바퀴 정지시간·실측 yaw가 아니다. 잘못된 행이나 필수 필드 누락은
따로 표시하고, 시간 역전이 있으면 관련 시간 통계를 무효화한다.

## 6. 되돌리기

새 tracker를 수동 disarm하고 새 컨테이너의 ROS 실행을 모두 종료한다. 새 컨테이너만
`docker stop moai_jackal_spubert_stability`로 종료한다. 기존 컨테이너를 보존했다면
기존 shell/실행 절차로 돌아간다. 기존 컨테이너가 없다면 audit에 기록한 원래 이미지
또는 snapshot과 원래 mount/DDS 설정으로 다시 만든다. 어떤 경우에도 기존과 새 tracker를
동시에 시작하지 않는다. 원래 이미지 tag, audit, calibration/maps/logs를 삭제하지 않는다.

이번 스크립트들은 기존 이미지 tag/소스/컨테이너를 자동 교체하지 않는다. 만들어진
새 보관용 base tag와 overlay tag의 제거도 사용자가 되돌리기 필요성을 확인한 뒤 결정한다.
