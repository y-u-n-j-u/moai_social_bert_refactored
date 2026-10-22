# 자칼 주행 절차서 (명령어 포함)

2026-10-12 기준. 이 문서의 명령은 그날 실제로 입력해서 쓴 것들이다. 직접 확인하지 못한 부분은 해당 위치에 **(미확인)** 으로 표시했다.

## 0. 시작하기 전에

- 항상 **e-stop 담당자가 로봇 옆에 있어야** 한다. e-stop이 모든 소프트웨어 정지보다 우선한다.
- 로봇은 `주행(RUN)` 칸을 누르기 전까지 움직이지 않는다. 그 전까지는 모든 명령이 준비 단계다.
- 호스트 PC(노트북)에서 하는 일과 컨테이너 안에서 하는 일이 섞여 있다. 아래 표기를 따른다.
  - `[호스트]` 일반 터미널, `[컨테이너]` 컨테이너 안(`docker exec -it moai_jackal_spubert bash ...`)
- 컨테이너는 `--rm`으로 만들어져서 **꺼지면 안에서만 바꾼 파일이 사라진다.** 끄기 전에 이미지로 저장한다(7절).
- 로그는 컨테이너의 `/root/jackal_logs/drive_latest`(가장 최근 실행 폴더로 연결된 링크)에 쌓인다.

## 1. 가장 쉬운 방법: 터미널 10칸 스크립트

```bash
[호스트]  ~/capstone_navi/real_jackal/field_scripts/open_jackal_terminals.bash
```

terminator 창 하나가 10칸으로 나뉘어 열린다. **각 칸에는 명령이 이미 입력되어 있고 Enter를 눌러야 실행된다**(고쳐서 실행해도 된다). 6번 STATUS만 자동으로 돈다(읽기 전용).

| 칸 | 이름 | 하는 일 |
|---|---|---|
| 1 | HOST | 네트워크 설정(sudo) + 컨테이너 실행. 이미 켜져 있으면 "already running"이라고만 하고 끝난다 |
| 2 | SENSORS | 라이다 드라이버, 10초 뒤 LIO(FAST-LIVO, 영상 꺼짐) |
| 3 | NAV2 + BRIDGE | Nav2(지도, AMCL, 플래너)와 브릿지/트래커 |
| 4 | RVIZ | 녹화용 RViz (오른쪽에 열리도록 설정됨) |
| 5 | ODOM TRAIL | 지나온 위치를 파란 선으로 그림. 주행이 시작되면 이전 선을 지움 |
| 6 | STATUS | 부하, FAST-LIVO, 프로세스, 브릿지/트래커 최근 줄 |
| 7 | CHECK | AMCL 안정화와 수치, 자칼 연결, 센서 주파수, 부하 점검 |
| 8 | RUN (!!) | **로봇이 움직인다.** AMCL을 다시 확인하고 rosbag을 시작한 뒤 주행 |
| 9 | STOP | 즉시 정지(stop) |
| 10 | AFTER | 주행 후 rosbag 정상 종료 |

옵션: `--only 이름`(한 칸만, 이름은 host sensors nav rviz trail status check run stop after), `--dry-run`(열지 않고 확인).
창 배치: 터미널은 화면 왼쪽 46%, RViz는 오른쪽에 열린다. GNOME(Wayland)이 창 위치 요청을 가끔 무시하므로, 겹치면 터미널에서 `Super + ←`, RViz에서 `Super + →`를 누른다.
**이미 센서/Nav2가 켜져 있으면 1~3번 칸을 다시 누르지 않는다**(노드가 중복된다).

## 2. 처음 켤 때 (부팅 후 한 번)

1. 자칼 본체와 라이다 전원을 켜고 이더넷을 연결한다.
2. 네트워크 주소를 추가한다(재부팅이나 케이블 분리 시 사라진다). sudo 비밀번호가 필요하다.
   ```bash
   [호스트]  sudo bash ~/capstone_navi/real_jackal/scripts/configure_jackal_network.bash up
   ```
   `[PASS]` 5개가 나오면 정상이다. 점검만 하려면 `... configure_jackal_network.bash check`.
3. RViz를 컨테이너에서 띄우기 위한 화면 권한(재부팅 후 한 번):
   ```bash
   [호스트]  xhost +SI:localuser:root
   ```
4. 컨테이너를 실행한다(이미지 이름은 최신 `drive-NNNN`, 현재 `drive-1012`).
   ```bash
   [호스트]  cd ~/capstone_navi_stability_20260916 && \
     CAPSTONE_CALIB=$HOME/capstone_navi/spu_deploy_docker/context/calib \
     IMAGE=drive-1012 CONTAINER_NAME=moai_jackal_spubert MOAI_ENABLE_GUI=1 \
     ./real_jackal/docker/run_stability_container.bash
   ```
   - 경로를 `…`처럼 줄여 쓰면 `Calibration directory must contain …` 에러가 난다. 위 전체 경로를 그대로 쓴다.
   - 이미 같은 이름의 컨테이너가 떠 있으면 새로 실행하지 않는다(`docker ps`로 확인).
   - 이 스크립트는 stability 폴더에 있는 것을 **읽어서 실행만** 한다. 그 폴더는 수정하지 않는다.
5. 컨테이너 안으로 들어가는 방법(칸마다 이 방식으로 연다):
   ```bash
   [호스트]  docker exec -it moai_jackal_spubert bash --noprofile --norc
   [컨테이너] source /root/jackal_runtime/scripts/jackal_ros_env.bash
   [컨테이너] source /root/moai_stability_ws/install/setup.bash
   ```

## 3. 센서와 Nav2 켜기 (컨테이너 안)

```bash
[컨테이너] bash /root/start_livox_only.bash      # 새 실행 폴더(drive_YYYYMMDD_HHMMSS)를 만들고 라이다 드라이버 시작
[컨테이너] sleep 10; bash /root/start_lio_fixed.bash   # LIO(FAST-LIVO, 영상 꺼짐), /scan 변환, 메모리 감시
```
약 1분 뒤 확인: `/livox/lidar`, `/aft_mapped_to_init`, `/scan` 이 모두 10Hz여야 한다.
```bash
[컨테이너] ros2 topic hz /aft_mapped_to_init --window 10
```
```bash
[컨테이너] bash /root/start_nav2_bridge.bash      # Nav2 시작, 8초 뒤 브릿지/트래커 시작 (약 40초)
```
브릿지 상태가 `hold:waiting_for_goal`이면 정상이다(아직 goal이 없다는 뜻).

## 4. 매 주행 (사람이 하는 단계 포함)

```bash
[컨테이너] bash /root/start_rviz.bash             # RViz (녹화용 배치). 궤적이 필요하면:
[컨테이너] python3 /root/odom_trail.py
```
1. 로봇을 시작 위치에 두고 박스를 배치한다. 배치를 기록해 둔다(주행 분석에 필요하다).
2. **RViz에서 `2D Pose Estimate`**: 로봇이 보는 방향으로 끌어서 찍는다(방향이 틀리면 AMCL 방향 오차가 커진다).
3. **리모컨(RC)으로 로봇을 1~2m 앞으로** 움직인다(AMCL이 수렴하도록).
4. **RViz에서 `2D Goal Pose`** 로 목표를 찍는다. 이때 GP(안내점)와 후보점이 나타난다.
5. 점검한다(7번 CHECK 칸과 같은 내용).
   ```bash
   [호스트]  docker exec -i moai_jackal_spubert bash --noprofile --norc -s < ~/capstone_navi/real_jackal/field_scripts/check_ready.bash
   ```
   판정: AMCL 표준편차 x≤0.60 m, y≤0.20 m, 방향≤10° → `PASS`. 자칼 연결 `PASSED (ready)`. odom/scan 10Hz. FAST-LIVO CPU 약 120% 이하.
   `FAIL`이면 2번부터 다시 한다.
6. 컴퓨터가 렉이 걸리면 RViz를 닫는다(소프트웨어 렌더링이라 CPU 약 2코어를 쓴다).
7. e-stop 담당자가 준비되면 **주행 시작**:
   ```bash
   [컨테이너] bash /root/monitor_run.bash
   ```
   스크립트가 AMCL을 한 번 더 확인하고, rosbag을 시작하고, 주행을 시작해서 목표에 도달하면 `GOAL REACHED`를 찍고 끝난다.
   자동 정지 조건: FAST-LIVO 메모리 2000MB 초과, odom 80m 초과, map→odom 이동 1m 초과(`LOCALIZATION JUMP`), 브릿지 에러, 장애물 0.26m 미만.
   **`Ctrl+C`나 터미널 닫기도 자동으로 stop한다**(`trap`). 출력을 `| head`로 넘기면 SIGPIPE로 감시가 죽으므로 하지 않는다.
8. 이상하면 즉시:
   ```bash
   [컨테이너] ros2 service call /spu_bert/enable_motion std_srvs/srv/SetBool "{data: false}"
   ```
   (e-stop이 항상 우선이다.)
9. 끝나면 rosbag을 정상 종료한다. **기록 중인 rosbag 파일을 열지 않는다**(SQLite 잠금으로 기록이 깨진다).
   ```bash
   [컨테이너] pkill -INT -f "ros2 bag reco[r]d"; sleep 5
   [컨테이너] ls -dt /root/jackal_logs/drive_latest/bag_[0-9]* | grep -v "\.log"
   ```
10. 로봇을 RC로 시작 위치에 돌려놓고 2번부터 반복한다.

## 5. 주행 후 분석

닫힌 rosbag(`/root/jackal_logs/drive_latest/bag_HHMMSS`)에서 아래를 본다. Claude에게 "분석해줘"라고 하면 같은 지표를 계산해 준다.

- 목표까지의 시간과 거리, 정지 합계와 가장 긴 정지(사유 포함), 10m당 회전량, 강한 회전(|w|>0.4) 시간, 회전 방향 전환 횟수
- 경로 출처(모델 / 유지 / 안전 보조 경로 `FB`), 경로 발행 간격(트래커 만료 2.5초 대비)
- GP: 로봇~GP 거리, GP~장애물 여유, GP가 1m 넘게 움직인 횟수
- 비교 기준: 장애물 구간을 정지 없이 통과한 주행은 10m당 회전량 약 166~189°, 정지 0~1.7초였다.

## 6. 설정/코드를 바꾼 뒤 반영 (컨테이너에는 직접 복사)

```bash
[호스트]  P=~/capstone_navi/real_jackal/ros2_ws/src/moai_jackal_spubert; D=/root/moai_stability_ws/src/moai_jackal_spubert
[호스트]  docker cp $P/config/real_jackal_social005.yaml moai_jackal_spubert:$D/config/real_jackal_social005.yaml
[호스트]  docker cp $P/moai_jackal_spubert/real_jackal_spubert_bridge_node.py moai_jackal_spubert:$D/moai_jackal_spubert/real_jackal_spubert_bridge_node.py
```
그 뒤 브릿지만 다시 시작한다(**goal이 사라지므로 `2D Goal Pose`를 다시 찍어야 한다**).
```bash
[컨테이너] R=$(readlink -f /root/jackal_logs/drive_latest)
[컨테이너] ros2 service call /spu_bert/enable_motion std_srvs/srv/SetBool "{data: false}"
[컨테이너] pid=$(awk '$1=="spubert_real"{print $2}' $R/pids | tail -1); kill -INT -- -$pid; sleep 6
[컨테이너] G=$(date +%H%M%S)
[컨테이너] setsid nohup ros2 launch moai_jackal_spubert real_jackal_spubert.launch.py cmd_vel_topic:=/j100_0519/cmd_vel use_cuda:=true require_global_path:=true launch_rviz:=false > $R/spubert_real_$G.log 2>&1 < /dev/null &
[컨테이너] echo "spubert_real $!" >> $R/pids
```
설정이 반영됐는지 확인: `ros2 param get /safe_path_tracker path_timeout_sec` 등. 코드를 바꾸면 먼저 `python3 -m pytest -q test`(저장소의 `moai_jackal_spubert/test`)를 돌린다.

## 7. 꺼야 할 때

- **LIO만 재시작**(FAST-LIVO CPU가 약 250% 이상이거나 약 1시간마다. 재시작하면 odom이 0으로 초기화되므로 브릿지를 다시 시작하고 포즈, RC, goal을 처음부터 다시 한다):
  ```bash
  [호스트]  docker exec -i moai_jackal_spubert bash --noprofile --norc -s < ~/capstone_navi/real_jackal/field_scripts/restart_lio.bash
  ```
  (미확인: 이 스크립트는 문법만 확인했고 실제 로봇 상태에서 돌려 보지 않았다.)
- **전부 끄기**(컨테이너는 유지):
  ```bash
  [호스트]  docker exec -i moai_jackal_spubert bash --noprofile --norc -s < ~/capstone_navi/real_jackal/field_scripts/stop_all.bash
  ```
  **실행 중인 주행 스택을 모두 끈다.** 다른 사람이 쓰는 중인지 확인하고 실행한다.
- **컨테이너를 끄기 전에 현재 상태를 이미지로 저장**한다(컨테이너 안에서만 바꾼 파일이 이 이미지에 남는다).
  ```bash
  [호스트]  docker commit moai_jackal_spubert drive-$(date +%m%d)
  ```

## 8. 문제와 해결

| 증상 | 원인과 해결 |
|---|---|
| `Calibration directory must contain …: …` | 명령의 `CAPSTONE_CALIB=`를 `…`로 줄여 쓴 것. 4절의 전체 경로를 쓴다 |
| 10칸 대신 평범한 terminator만 뜬다 | 이미 떠 있는 terminator가 새 창을 가로챔. 스크립트는 `--no-dbus`로 독립 프로세스를 띄운다 |
| 주행이 멈췄다 갔다 한다 | (1) 컴퓨터 과부하: LIO 재시작, RViz 닫기 (2) 경로 만료 시간이 너무 짧음: 현재 2.5초 |
| FAST-LIVO CPU가 250%를 넘는다 | 지도가 커짐. LIO를 재시작한다 |
| 감시 스크립트가 `NOT STARTING - AMCL not converged` | 포즈를 다시 찍고 RC로 1~2m 움직인다 |
| 브릿지가 `waiting_for_goal` | goal이 없다. `2D Goal Pose`를 찍는다(브릿지를 재시작하면 항상 필요) |
| `hold:adaptive_guidance_failed` | 안내점을 못 찾음(장애물이 코앞). 로봇을 조금 뒤로 빼거나 박스 위치를 확인한다 |
| `LOCALIZATION JUMP > 1 m -> STOP` | AMCL이 1m 넘게 튀어서 자동 정지. 포즈를 다시 찍고 처음부터 |
| rosbag이 비었거나 깨졌다 | 기록 중에 파일을 열었거나 종료 전에 컨테이너를 끈 경우. `SIGINT`로 종료한 뒤에만 연다 |
| `pkill -f '패턴'`을 쳤더니 내 셸이 종료된다 | 패턴이 내 명령줄에도 있어서 자기 자신이 걸린다. `pkill -f '[p]attern'`처럼 대괄호를 쓰거나 pid로 종료한다 |

## 9. 화면에 나오는 것

- 초록 굵은 선: 로봇이 실제로 따라가는 경로. 파란 선: 지나온 궤적. 초록 점: 노란 안내점(GP) 1m 안의 안전한 모델 후보.
- 흰 글씨 `FB`: **안전 보조 경로가 지금 실행 중**이라는 표시다(모델이 경로를 만들 때는 뜨지 않는다). 영상이나 발표에 쓸 때는 슬라이드에 "FB = 안전 보조 경로"라는 범례를 넣는다.
- 거절된 후보와 충돌 검사 표시는 RViz에서만 숨겨 있고, 로그와 rosbag에는 그대로 남는다.

## 10. 이름 정리

컨테이너 `moai_jackal_spubert`, 이미지 `drive-1012`(최신), `drive-1007`, `moai-jackal-spubert:drive-1006` 등, 로그 링크 `drive_latest`.
(이전에 쓰던 다른 이름의 이미지와 링크는 같은 것에 새 이름표를 붙인 것이다. 옛 로그 링크가 남아 있어도 스크립트는 쓰지 않는다.)

## 11. 보행자 인식 시험 (0단계: 로봇 정지, 인식만, 2026-10-22)

로봇을 움직이는 노드(트래커, 브릿지)와 FAST-LIVO, Nav2는 올리지 않는다. 부하가 거의 없다(load 약 1).

```bash
[호스트]  xhost +SI:localuser:root          # 재부팅 후 한 번. 빠지면 RViz가 안 뜬다("Authorization required")
[호스트]  ~/capstone_navi/real_jackal/field_scripts/open_jackal_terminals.bash --ped
```
7칸: 1 HOST(네트워크 sudo + 컨테이너, 이미 켜져 있으면 건너뜀) / **2 PED SENSORS**(라이다, 카메라, YOLO, 3D 투영, 추적기, 표시 노드. 약 1분) / **3 PED RVIZ** / 4 PED STATUS(자동) / **5 PED RECORD**(한 번만) / 6 PED STOP RECORDING / 7 KILL ALL.

- 기록은 **카메라 원본 영상 포함, 약 1.1GB/분**이다(압축 영상 토픽이 이 환경에 없다). 기록 중인 파일은 열지 않는다. 5번을 두 번 눌러도 두 번째는 건너뛴다.
- 분석: 6번으로 기록을 끝낸 뒤 Claude에게 "분석해줘". 거리별 YOLO 검출, 3D 위치의 라이다 점 지지, 추적 ID/속도, 오검출을 계산한다.
- RViz 표시: 3D 박스와 추적, **빨간 원 1.2m**(후보 경로 탈락 거리), **주황 원 0.9m**(트래커 정지 거리), 하늘색 4.8초 예측선, 라벨 `id v 추적기 / 브릿지`.
- 정리: `~/capstone_navi/real_jackal/field_scripts/kill_all.sh`.

### 0단계 결과 (짧은 기록 1개, 사람이 보인 시간 약 11초)
- 잘 됨: YOLO 점수 0.91~0.97, 매 프레임 검출, 3D 위치 손실 없음, 추적 위치에 라이다 점 있음(24개 중 20개), 사람이 없는 구간 오검출 0건, 보이는 동안 ID 유지.
- 문제: 사람이 아주 가까워지거나 시야를 벗어나면 추적이 **유령**이 된다(실제 0.5~1.2m인데 추적 3.3~4.2m, 속도 2.41m/s). 가까울 때 속도 추정이 흔들린다. 3m 넘는 거리 데이터는 아직 없다.
