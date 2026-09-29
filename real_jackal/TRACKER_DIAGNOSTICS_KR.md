# 추종점·진단 검증 안내

## 적용 범위와 한계

- 추종점을 가장 가까운 선분 위 투영점에서 경로 진행 방향으로 계산한다.
  로봇→가장 가까운 정점 거리나 횡방향 오차를 lookahead 거리에서 빼지 않는다.
- 같은 경로 형상·frame에서는 투영 진행 거리가 감소하지 않는다. 최초 획득 후에는
  이전 진행 거리부터 `로봇 변위 + lookahead 거리`만큼 앞까지 투영을 검색하여
  교차점에서 먼 미래 구간으로 건너뛰는 것을 제한한다. 이는 전진 경로용이며
  후진·재위치추정 복구 알고리즘은 아니다.
- 새 형상·frame, 빈 경로를 수신하면 진행 이력을 초기화한다. 동일 형상의 재발행은
  수신 시각만 갱신하고 진행을 초기화하지 않는다. 새 경로 선택 자체는 바꾸지 않는다.
- 퇴화/비유한 경로는 정지한다. 목표에 도달하지 않았어도 짧은 경로를 소진하면
  `path_endpoint_reached`로 정지하고 새 유효 경로를 기다린다.
- 빈 경로·센서 누락·기존 안전거리·무장·목표 도착 조건은 유지한다. 새 안전 검사에
  실패한 경로를 유지하거나 timeout을 늘리는 변경은 없다.
- 센서 시각 정렬, 저속 heading, 후보 연속성, 도착 상태 공유, 조향 튜닝은 별도 작업이다.
  이 수정만으로 실차 요잉·반복 정지 전체가 해결됐다고 볼 수 없다.

## 자칼 없이 시험

호스트에서 실행한다. ROS를 source하지 않으며 DDS, 센서, 로봇 명령을 사용하지 않는다.

```bash
cd /home/moai/capstone_navi
PYTHONDONTWRITEBYTECODE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH="$PWD/real_jackal/ros2_ws/src/moai_jackal_spubert" \
python3 -B -m pytest -q -p no:cacheprovider \
  real_jackal/ros2_ws/src/moai_jackal_spubert/test
```

시험에는 기존 오류의 2cm 경계 사례, 횡방향 오차, 곡선, 중복점, 끝점, 교차 경로,
새 경로 초기화 및 실제 tracker 메서드의 안전 정지·진단값 검증이 포함된다.
`test_tracker_offline.py`는 메모리 내 ROS 대역을 사용하므로 DDS/실제 메시지
타입 지원이나 실차 성능까지 검증하는 시험은 아니다.

2026-09-15 검증 결과: 오프라인 시험 51개 통과. 별도로 기존 Humble 이미지에서
`--network none`, 장치 마운트 없음, 읽기 전용 소스, ROS domain 231을 사용하는
임시 컨테이너로 새 노드를 검증했다. 실제 ROS의 JSON String·Marker·dry-run Twist
수신, `(0.39, 0.05)`에서 추종점 `(1.09, 0)`, 빈 경로 즉시 정지를 확인했다.
주행 컨테이너나 배포 이미지를 갱신한 검증은 아니며 실차 효과는 미검증이다.

## 컨테이너 반영 주의

현재 Dockerfile은 Python 패키지를 이미지에 COPY한다. 호스트 소스 수정만으로
이미 만들어진 이미지/컨테이너가 갱신되지는 않는다. 수정된 패키지를 반영한 프로젝트
이미지로 재빌드·재생성하거나, 주행을 정지시킨 뒤 명시적으로 패키지를 배포해야 한다.
실행 중인 tracker를 덮어쓰거나 다른 팀의 이미지·서비스·`.bashrc`를 수정하지 않는다.

반영 뒤 컨테이너에서 아래 명령으로 새 코드가 import되는지 확인할 수 있다.
이 명령은 노드를 실행하거나 무장하지 않는다.

```bash
python3 -c 'import moai_jackal_spubert.tracker_diagnostics as d; print(d.__file__)'
```

## 진단 확인

수정본 tracker가 실행 중인 컨테이너의 진단 터미널에서 읽기만 한다.

```bash
ros2 topic echo /spu_bert/tracker_diagnostics std_msgs/msg/String --field data
```

진단은 JSON 문자열이며 제어 주기마다(기본 10Hz), 추가 정지 콜백에서도 발행한다.
기존 `tracker_status`는 호환성을 위해 그대로 유지되며 텍스트 변경 시만 발행한다.
진단 추가를 위해 제어 콜백에서 파일 쓰기는 하지 않는다. 장기 기록은 별도로
rosbag 등으로 수집해야 한다.

| 필드 | 의미 |
| --- | --- |
| `stamp_ns`, `schema_version` | 진단 시각과 형식 버전 |
| `event`, `status` | 제어 실행/정지 및 현재 사유. `tracking`도 제자리 회전일 수 있음 |
| `motion_enabled`, `emergency_latched` | 현재 내부 무장·소프트웨어 비상정지 latch 상태 |
| `path.geometry_id` | 이 노드에서 관측한 형상/frame 변경 번호. 모델 후보 ID나 영구 plan ID 아님 |
| `path.receipt_id` | 동일 경로 포함 수신 횟수 |
| `path.present`, `path.fresh` | 경로 보유 여부·기존 수신시각 timeout 충족 여부. 충돌 안전 보증 아님 |
| `path.stamp_ns`, `source_age_s`, `received_age_s` | 발행 헤더 시각, 헤더 기준 나이, 수신 후 시간 |
| `lookahead.projection`, `target` | 경로 투영점과 실제 추종점. `path.frame_id` 좌표계 |
| `segment_index`, `target_segment_index` | 원본 경로의 투영·추종 선분 번호(0부터) |
| `progress_m`, `target_progress_m`, `path_length_m` | 경로 시작부터의 호 길이 |
| `cross_track_error_m`, `heading_error_rad` | 횡방향 거리·조향 방향 오차 |
| `search_start_m`, `search_end_m` | 진행 연속성을 위한 투영 검색 범위 |
| `target_command` | 가감속 제한 전 목표 `v`(m/s), `w`(rad/s) |
| `published_command` | 이 tracker가 실제 발행한 제한 후 Twist. mux 출력·바퀴 실측과 다름 |
| `odom_reported_twist` | odom 메시지가 보고한 속도. 해당 발행자가 유효하게 채우는지 별도 확인 필요 |
| `inputs` | odom/scan 측정 헤더·수신 시간, 보행자 수신 후 시간 |
| `stop_episode.first_reason` | 연속 정지 구간에서 tracker가 처음 내린 사유 |
| `stop_episode.last_completed` | 직전 정지 구간의 최초 사유·지속시간 |

`planner_hold → predicted_path_missing_or_stale`이면 최초 사유는 `planner_hold`로
유지된다. 이는 **tracker의 최초 사유**이며 bridge의 구체적인 후보 탈락 원인은
기존 bridge 진단 JSONL과 함께 확인해야 한다. 정지 시간은 소프트웨어 상태 시간이지
바퀴 정지 실측 시간이 아니다. `status`와 `emergency_latched`를 함께 확인한다.

무한대·누락된 진단값은 JSON `null`이다. source age가 음수이면 시계/타임스탬프를
점검해야 한다. 이번 수정은 시각을 기록할 뿐 기존 freshness 판단을 교체하지 않는다.

## RViz

수정된 RViz 설정에는 `Tracker Lookahead (orange)`를 추가했다. 기존 RViz 세션이라면
`Add → Marker`, Topic `/spu_bert/tracker_lookahead`를 직접 추가할 수 있다.
주황색 구는 tracker가 선택한 추종점이며 MGP 목표 후보가 아니다.
정지 시 삭제하고 0.5초 lifetime을 적용해 오래된 점이 계속 남지 않도록 했다.
비무장·입력 누락 등 제어가 정지 상태이면 추종점은 표시하지 않는다.

## 다음 실차 시험

기존 물리 E-stop·주행 허용 절차를 그대로 지킨다. 먼저 정지 상태 입력 검증,
분리된 dry-run 출력, 고정 경로 검증을 하고 이후 통제된 저속 시험으로 진행한다.
진단 결과를 보려고 실주행 토픽으로 무장해서는 안 된다. 현재 무장 상태는 과거
`Motion armed` 응답이 아니라 새 진단의 `motion_enabled`와 안전 상태로 확인한다.

## 충돌 판정 진단 (2026-09-15 추가)

검증: 전체 오프라인 테스트 81개 통과. 기존 51개에 충돌 샘플/판정 보존,
현재 위치·중간 선분·미관측 영역 충돌, bridge 전체 후보 탈락→빈 경로→유효 경로 재개,
진단 실패 시 정지 유지, JSON/Marker/요약 CLI 테스트를 추가했다. 별도의 네트워크 없는
임시 컨테이너에서 실제 ROS Humble MarkerArray 구/텍스트 수신·2초 lifetime·삭제·JSON
직렬화도 확인했다. 모델 추론/실차 주행/장시간 처리 지연을 검증한 것은 아니다.
실행 중 배포 컨테이너와 기존 이미지 태그는 변경하지 않았다.

이 추가는 `robot_footprint_collision`의 관측을 위한 것이다. 충돌 샘플·반경·검사 순서,
후보 선택, 안전 정지 조건, 모델 및 제어 게인은 변경하지 않는다. 빈 경로 정지를
진단 파일 기록보다 먼저 수행하며, 파일/Marker 출력 실패는 경고로 남긴다.

bridge의 기존 `prediction` JSONL에 다음이 추가된다. 기존 필드는 유지한다.

- `collision_context`: 검사 map frame, 로봇 `[x,y,yaw]`, 로봇 반경/여유/합산 반경,
  map 해상도·원점·크기·미관측 영역 정책, map 갱신 시 사용한 scan/odom의 header 시각,
  header 나이와 수신 나이. 현재 구현은 최신 odom과 최신 scan을 조합하며,
  **scan 시각의 자세로 정렬한 결과가 아니다**. 이 정책도 명시적으로 기록한다.
- `attempts[].path`: 선택되지 않은 후보도 포함한 좌표. 비유한 값은 JSON null이다.
- `attempts[].collision.swept_check_performed`: 최종 swept 검사를 실제 수행했는지.
  모델 단계에서 먼저 탈락한 경우 false이며, 이는 충돌이 없다는 뜻이 아니다.
- `current_footprint_collision`: 현재 위치의 로봇 반경 검사 결과. 검사 안 함은 null.
- `first_collision_point`: 최초 충돌 판정 **로봇 중심 좌표**, 장애물 셀/접촉점이 아님.
- `first_collision_segment_index`: 0은 현재 위치→path[0], 1은 path[0]→path[1].
  현재 위치 자체에서 탈락한 경우 null이며 sample index와 distance는 0이다.
  경계점은 앞쪽 선분(먼저 검사한 선분)에 속한다.
- `first_collision_distance_m`: 현재 위치에서 해당 검사 중심까지 경로를 따른 거리.
- `footprint_collision_count`, `swept_sample_count`, `sample_spacing_m`: 기존 판정의
  충돌 샘플 수/전체 샘플 수/최대 보간 간격. count는 장애물 개수가 아니다.

이 기록만으로 occupied 셀과 unknown/out-of-map을 구별하거나 map 전체를 복원할 수는 없다.
필요하면 `/spu_bert/debug_rolling_map`, `/scan`, `/aft_mapped_to_init`을 함께 rosbag에
기록한다. 기존 시각의 로그에는 새 진단이 소급해 생기지 않는다.

### RViz

새 설정의 `Collision Check Centers (red-magenta)` 또는
`Add → MarkerArray → /spu_bert/collision_markers`를 사용한다.

- 빨간 구: 경로 상의 최초 충돌 판정 중심.
- 자홍색 구: 현재 로봇 위치 자체에서 충돌 판정.
- 흰 글씨: 후보 rank와 선분 번호. 여러 후보의 중심이 같으면 구가 겹칠 수 있다.
- 다음 예측에 기존 마커를 지우고 갱신하며, 2초 후 자동 만료한다.
  sensor/route 등의 hold에서도 지운다. 프레임은 검사에 사용한 odom frame이다.
- 후보가 전부 탈락한 경우뿐 아니라 일부만 탈락한 경우도 표시한다.
  따라서 빨간 구가 있다고 반드시 tracker가 정지한 것은 아니다.

### 짧은 요약 명령

컨테이너에서 실행한다. 읽기 전용이며 ROS 연결이 필요 없다.

```bash
python3 /root/jackal_runtime/scripts/summarize_collision_diagnostics.py \
  /root/jackal_logs/spubert_real_jackal_diagnostics.jsonl --last 5
```

`--since 1789471672 --until 1789471678`처럼 ROS 시각(초) 범위를 지정할 수도 있다.
`collision_diagnostics_available: false`이면 이전 형식 로그이다.

### 배포 및 검증 순서

소스 수정만으로 실행 중 컨테이너는 바뀌지 않는다. 기존 `lookahead-v1`을 보존하고
`collision-diag-v1` 등 별도 이미지 태그로 빌드한 뒤 **IMAGE 환경변수로 그 태그를
명시**하여 계획된 정지 상태에서 컨테이너를 재생성한다. 다른 팀 설정과 NUC 서비스는
수정하지 않는다. 재생성 전에 컨테이너 내부에만 있는 로그/파일을 보존한다.

새 이미지에서 아래 import를 먼저 확인한다.

```bash
python3 -c 'import moai_jackal_spubert.collision_diagnostics as d; print(d.__file__)'
```

물리 E-stop, dry-run 출력 분리와 tracker 비활성화를 유지하고 센서·위치 추정·Goal을
준비한다. **충돌 진단은 bridge가 생성하므로 tracker를 활성화하지 않아도 기록된다.**
같은 Goal에서 valid→invalid→valid 전환을 수집하고, 최초 충돌 중심이 실제 벽/장애물
또는 미관측 영역과 겹치는지 확인한 뒤 다음 수정 범위를 결정한다.
