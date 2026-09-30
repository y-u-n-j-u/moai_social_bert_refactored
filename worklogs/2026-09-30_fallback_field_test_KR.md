# 2026-09-29~30 작업 정리 및 fallback 현장 테스트 절차

## 1. 지금까지 한 일

### 코드 (`capstone_navi`, 브랜치 `jackal-real-drive-fixes-20260912`)
| 커밋 | 내용 |
|---|---|
| `f6d501c` | Nav2 `robot_radius` 0.5 정렬 + collision/tracker 진단 도구, offline sensitivity 분석, 테스트 추가 |
| `1e651b2` | `robot_radius`를 다시 0.34로 되돌림 + lateral-detour fallback(`guided_spubert_runtime.lateral_guidance_point`) 추가 |

- 두 커밋 모두 origin에 push 완료. pytest 96개 통과.
- **robot_radius 교훈:** Nav2 global_costmap `robot_radius`를 브릿지 footprint 컷오프(0.34 + 0.16 = 0.5)에 맞추면, 폭 약 2m 복도에서 NavFn이 `failed to create plan with tolerance 0.50`으로 계획을 못 만든다(2026-09-29 현장). 0.34를 유지하고, 브릿지의 0.5m footprint 검사와 lateral-detour fallback이 안전을 담당한다.
- 해결하려던 문제: 2026-09-21 로그에서 같은 chokepoint에서 약 167초, 105초 연속 hold (`all_candidates_rejected` / `adaptive_guidance_failed`, 후보가 전부 footprint 검사에서 탈락하고 fallback이 없었음).

### 두 코드 갈래 (아직 합치지 않음)
- `capstone_navi` (junwoo 계열): blocked_hold/recovery, collision 마커, `PathProgressTracker`
- `capstone_navi_stability_20260916` (`jackal-stability-20260916`, HEAD `5a9900b`): plan_context, phase timing, GoalLifecycle, route_progress, route 변환 캐싱
- stability로 `f6d501c`를 cherry-pick하면 브릿지·tracker(각 10군데), `navigation_core.py`(3군데)에서 충돌한다. 기준을 먼저 정하고 필요한 기능만 옮긴다. stability 폴더는 변경하지 않고 원상태로 두었다.

### Docker 이미지
- 새 이미지: `moai-jackal-spubert:fallback-1e651b2` (22.5GB)
  - 베이스 `moai-jackal-spubert:collision-diag-v1` (sha `ca07002…`) 위에 `1e651b2` 패키지를 `/root/moai_stability_ws`로 오버레이
  - 빌드 중 매니페스트 검증, CPU 모델 smoke test 통과. 이미지 안에서 `lateral_guidance_point` 존재, `robot_radius: 0.34`, 소스 해시 일치 확인
  - 빌드만 했고 컨테이너/ROS 노드는 시작하지 않음
- **검증된 이미지는 계속 `moai-jackal-spubert:stability-20260921-continued`.** 새 이미지에는 stability 수정(route 캐싱, route_progress 등)이 없으므로 fallback 동작 확인용으로만 쓴다.

## 2. 현장 테스트 절차 (fallback 검증)

**목적:** 9/21 chokepoint에서 전 후보가 거부될 때 lateral-detour로 우회하는지, 그리고 `robot_radius` 0.34에서 Nav2가 계획에 실패하지 않는지 확인한다.

**원칙:** 물리 E-stop 담당자가 옆에 있는 상태에서만 arm한다. 매 arm 전에 `2D Pose Estimate`로 다시 localize한다(재localize 없이 두 번째 테스트를 하면 AMCL이 이탈해 `map->base_link`가 튄 적이 있음).

### 0. 사전 준비
- [ ] 디스크 여유 확인 (`df -h /`, 지금 41GB, 사용률 96%)
- [ ] 이전 세션의 ROS 프로세스가 남아 있지 않은지 확인. `ros2 launch` wrapper를 죽여도 자식 노드가 남는 경우가 있다. `filterpass.py`, `pointcloud_to_laserscan_node`는 `full_stack.launch.py`의 자식이 아니라 별도 프로세스다.
- [ ] 노트북 UDP 수신 버퍼 설정 유지 확인 (`/scan` 지터의 원인이었음. `rmem_default` 변경 후 ROS 파이프라인을 한 번 재시작해야 반영됨)
- [ ] `docker ps -a`로 컨테이너 이름 충돌 확인 (이번에는 새 이름 사용)

### 1. 컨테이너 기동 (검증된 컨테이너와 분리)
```bash
cd ~/capstone_navi_stability_20260916
IMAGE=moai-jackal-spubert:fallback-1e651b2 \
CONTAINER_NAME=moai_jackal_spubert_fallback \
CAPSTONE_CALIB=<calib 디렉터리 (extrinsic.txt, intrinsic.txt 포함)> \
MOAI_ENABLE_GUI=1 DISPLAY="${DISPLAY:-:0}" \
./real_jackal/docker/run_stability_container.bash
```
- 이미지에 `org.moai.stability.overlay=true` 라벨이 있어야 한다(있음). 기본 CMD는 `sleep infinity`라서 컨테이너가 떠도 아무 노드도 실행되지 않는다.
- `CAPSTONE_CALIB` 호스트 경로는 이 컴퓨터에서 확인하지 못했다. 기존 컨테이너를 만들 때 쓴 경로를 그대로 사용한다.
- 기존 검증 컨테이너는 롤백용으로 유지한다. 이름이 다르므로 서로 건드리지 않는다.

### 2. 센서·localization 스택 (기존 runbook 순서)
1. `ros2 launch realsense2_camera rs_launch.py` (`lsusb`에 `8086:0b5c`가 보여야 함. 빠지면 물리적으로 다시 꽂는다)
2. Livox MID-360 드라이버 노드 (runbook 참고)
3. `ros2 launch mid360_bringup full_stack.launch.py use_fast_livo:=true fast_livo_odom_fallback:=false extrinsic_path:=/root/data/calib/extrinsic.txt`
4. `ros2 run mola_bringup filterpass.py`
5. `ros2 run pointcloud_to_laserscan pointcloud_to_laserscan_node` (cloud_in → `/livox/lidar_filtered`, scan → `/scan` remap)
6. `/scan`이 나오는지 확인. AMCL은 `/scan` 없이는 `map->base_link` TF를 내지 않는다.
7. `ros2 launch moai_jackal_spubert nav2_route_planner.launch.py map:=/root/jackal_maps/world_map.yaml`
8. RViz `2D Pose Estimate` 후 로봇을 수십 cm 움직여(RC) AMCL을 수렴시킨다(x 표준편차 0.41 → 0.35m 확인한 적 있음). 정지 상태의 `/request_nomotion_update`만으로는 잘 수렴하지 않는다.
7. `ros2 lifecycle get /map_server`, `/amcl`, `/planner_server`가 모두 `active`인지 확인

### 3. Dry-run 먼저 (실제 cmd_vel 연결 금지)
```bash
ros2 launch moai_jackal_spubert real_jackal_spubert.launch.py \
  cmd_vel_topic:=/spu_bert/cmd_vel_dryrun use_cuda:=true \
  require_global_path:=true launch_rviz:=true
```
- 9/21 chokepoint 근처 goal을 지정하고 `/spu_bert/runtime_status`, `/spu_bert/tracker_status`를 본다.
- dry-run에서 `enable_motion`을 arm하고 `ros2 topic echo /spu_bert/cmd_vel_dryrun`으로 비영 속도가 나오는지 확인한 뒤 disarm한다. arm 전에 `ros2 topic info /spu_bert/cmd_vel_dryrun --verbose`로 구독자가 없는지 확인한다.
- 새로 확인할 것: 장애물 앞에서 후보가 전부 거부되는 상황에서 `all_candidates_rejected`가 오래 지속되지 않고 fallback 경로가 나오는지, Nav2가 `failed to create plan`을 내지 않는지.
- 5분 이상 rosbag을 남긴다(`/root/jackal_logs`). 컨테이너를 지워도 남는 마운트 경로다.

### 4. 실제 주행 (저속, 단계적)
```bash
/root/jackal_runtime/scripts/verify_jackal_link.bash preflight   # 실패 항목이 있으면 arm 금지
ros2 launch moai_jackal_spubert real_jackal_spubert.launch.py \
  cmd_vel_topic:=/j100_0519/cmd_vel use_cuda:=true \
  require_global_path:=true launch_rviz:=false
/root/jackal_runtime/scripts/verify_jackal_link.bash ready       # E-stop 해제 상태 요구
ros2 service call /spu_bert/enable_motion std_srvs/srv/SetBool "{data: true}"
```
- 시험 순서: 바퀴를 띄운 상태 → 0.25 m/s 직선 → 정적 장애물 → 보행자 1명 → 다중 보행자
- `/j100_0519/cmd_vel`은 twist_mux의 최저 우선순위 입력이다(joy 10, rc 12, e-stop 255가 더 높음). 경쟁 publisher가 없고 publisher/subscriber가 정확히 하나씩인지 확인한다.
- 즉시 disarm: `ros2 service call /spu_bert/enable_motion std_srvs/srv/SetBool "{data: false}"`
- 소프트웨어 e-stop(latch): `ros2 topic pub --once /spu_bert/emergency_stop std_msgs/msg/Bool "{data: true}"`. 다시 arm하려면 `SetBool false`로 latch를 reset한 뒤 `SetBool true`.
- 장애물에서 멈추면 RC로 조금 우회시킨 뒤 새로 `2D Pose Estimate`를 하고 재개한다(9/21에 두 번 이렇게 복구함).

### 5. 합격 기준 / 기록
- [ ] chokepoint에서 hold가 나도 수 초 안에 fallback으로 우회한다(이전: 167s, 105s)
- [ ] fallback 경로가 장애물 0.5m footprint 안으로 들어가지 않는다
- [ ] `failed to create plan` 없음
- [ ] 개방 구간에서 기존과 같이 hold 0회로 주행한다(회귀 없음)
- [ ] 로그: `/root/jackal_logs` 진단 jsonl과 rosbag 보관. 결과와 이미지 태그를 함께 기록한다

## 3. 그 다음
1. 테스트 결과에 따라 병합 기준 결정. fallback이 잘 되면 stability 쪽에는 `lateral_guidance_point`와 관련 부분만 옮기는 것이 가장 작은 변경이다.
2. fallback을 포함한 새 이미지를 stability 코드 기준으로 다시 빌드한다.
3. 정리: `capstone_navi`의 untracked `jackal-backup-*` 폴더, 깨진 ref `refs/codex/turn-diffs/...`

## 4. 참고 (Git)
- `y-u-n-j-u/moai_social_bert_refactored`에 push하려면 `gh auth switch --user wnsdnnn`이 필요하다. 기본 활성 계정 `minsong0206`은 403이 난다. push 후 다시 `minsong0206`으로 돌려놓는다.
