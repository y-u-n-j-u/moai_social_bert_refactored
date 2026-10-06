# 2026-10-06 실험 기록: 정적 장애물 회피 주행, 부하 정리, 남은 문제

앞선 기록(`2026-09-30_fixes_and_todo_KR.md`)의 후속이다. 코드는 `capstone_navi` 브랜치 `jackal-real-drive-fixes-20260912`의
`7488272`, `3d4c1e0`에 있다. 오프라인 테스트 153개 통과. 오늘 상태는 이미지 `moai-jackal-spubert:fallback-1006`으로 보존했다.

## 1. 요약

- 장애물(박스) 옆을 **정지 없이 통과해 goal(약 22m)에 도달**했다. 모델 후보가 그대로 쓰인 비율은 예측 40번 중 36번(90%).
- 막혔던 원인 하나를 확정했다: **로봇 자신의 출발 위치가 계획 반경(0.44m) 안**이라 모든 후보가 index 0에서 충돌로 판정(start-in-collision 교착).
- 장애물 앞에서 "멈추고 회전하는" 원인을 기록으로 찾았다: 모델 후보 순위가 바뀌며 경로가 좌우로 뒤집힘 + tracker가 방향 오차 0.95 rad에서 무조건 제자리 회전.
- 아직 해결 못한 것: (a) 복도 한가운데 장애물(틈 0.7~0.9m, 계획 한계선), (b) 주행 중 AMCL 보정으로 경로가 최대 1m 옆으로 이동, (c) **보행자 회피는 한 번도 실로봇에서 검증하지 못함**.

## 2. 센서 구성 (확인한 사실)

| 센서 | 연결 | 토픽 | 용도 |
|---|---|---|---|
| Livox MID-360 | 이더넷 (192.168.1.130) | `/livox/lidar` 10Hz | FAST-LIVO, `/scan`(높이 0.1~1.5m), 보행자 3D 추적, 롤링 지도 |
| Livox 내장 IMU | 동일 | `/livox/imu` 200Hz | FAST-LIVO 입력 |
| RealSense D455 | USB | 컬러 + camera_info | 보행자 YOLO 검출만 (VIO는 꺼짐) |
| Jackal 본체 | NUC (192.168.50.2) | `/j100_0519/cmd_vel` | 명령 출력. e-stop/모터 피드백/MCU 상태는 점검 스크립트만 읽음 |

- odom은 FAST-LIVO `/aft_mapped_to_init` **하나뿐**이다. Jackal 휠 오도메트리와 Jackal 자체 IMU는 구독하지 않는다.
- **IMU 단위:** `livox_ros_driver2`는 가속도를 g로 낸다(정지 시 크기 0.997). 9.80665배 수정은 `stability-20260921-continued`에만 있고
  `fallback-0930-fixed`/`fallback-1006`에는 없다. 정지 상태 odom은 안정적이었으나 이동 중 정확도 영향은 미측정.
- 깊이 스트림은 구독자 0명(미사용). 카메라와 YOLO 없이도 `/ped_tracking`이 10Hz로 유지되어 tracker heartbeat 요구는 충족된다.

## 3. 부하 분석 (정정 포함)

- **정정:** `ps -o pcpu`는 시작 이후 평균이라 실시간 값이 아니다. 앞서 "FAST-LIVO CPU 52→85%, RealSense 60%, ped_yolo 49%"라고 한 것은 평균값의 착시였다.
  실시간(`top`)으로는 FAST-LIVO 52%, RViz 28%, 브릿지 12%이고 호스트 load average는 카메라/YOLO 끈 뒤 8 → 1.7.
  `rsswatch.sh`를 `/proc` tick 차이로 실시간 CPU(`cpu_now`)를 기록하도록 고쳤다.
- FAST-LIVO 실시간 CPU는 22m 주행 뒤 52% → 120%, RSS 140 → 218MB로 늘었다. 주행 거리에 따라 커지므로 긴 주행 전 스택 재시작 권장.
- RealSense 깊이 끄기(`enable_depth:=false`)는 CPU를 거의 줄이지 못했다(컬러 1280x720 발행 비용이 큼). 정적 장애물 시험에는 카메라 자체가 불필요.
- **모드 구분:** `drive`(카메라/YOLO 없음, `full_stack ... launch_ped_yolo:=false`)와 `ped`(카메라 컬러 15Hz + YOLO, 깊이 없음).

## 4. 후보 거절 통계 (09-30 주행 450사이클, 후보 2035개)

유효 60%, `robot_footprint_collision` 27%, `insufficient_goal_progress` 6%, 목표점 탈락(`model_map_check_failed`+`no_map_safe_goal`) 5%.
모델 1순위 탈락 45%, 후보 전부 탈락 20%. stability의 MGP 표본 보존 수정(`fbff8b3`)은 목표점 탈락(약 5%)만 다루고
`robot_footprint_collision`(27%)은 다루지 않는다. 단일 cherry-pick은 stability의 heading/selection 설정과 얽혀 불가능(충돌 확인).

## 5. 주행 시험

| # | 상황 | 결과 |
|---|---|---|
| A | 박스 모서리 옆 통로 | 박스 모서리가 로봇 중심에서 ~0.35m(몸체 옆면 13cm). 계획 반경 0.44m라 **출발 위치부터 충돌 판정 → 모든 후보 탈락**. 그 뒤 recovery sweep(제자리 회전)이 모서리를 박스에 접근시킬 수 있었음 |
| B | 같은 박스, 적응형 탈출 반경 적용 | **정지 없이 goal 도달.** 모델 40번 중 36번 유효, fallback 10번(일반 5, 탈출 5) |
| C | goal 21.6m (직선) | **약 95초에 도달.** 멈춤/회전은 3곳: 박스 근처 감속 1.4초, 경로 공백 신호 1번, **12.6m 지점 제자리 회전 1.0초 + 감속 1.3초** |
| D | goal 약 35m, 옆 +4.8m | 처음부터 좌우로 꺾는 지그재그(w ±0.5, 모델 후보 순위가 1초마다 바뀜). `map→odom`이 arm 직후 0.06m에서 9m 주행 중 **0.95m**로 변함(AMCL이 pose 클릭 오차를 보정하는 중) |
| E | 복도 한가운데 장애물 | 로봇이 복도 방향에서 36° 틀어진 채 정면 0.65~1.2m에 장애물, 옆 틈 0.7~0.9m. `clearance_probe`는 로봇 자체는 전 반경에서 비충돌. 틈이 탈출 반경 요구 폭(0.72m)의 한계선이라 후보 전부 탈락 |

### C의 제자리 회전 원인 (rosbag `tracker_diagnostics`로 확인)
모델 1순위가 탈락한 사이클에 2순위 후보(옆으로 ~90° 휘는 0.6m 짧은 경로)가 선택됨 → lookahead 목표가 정면 `(13.13,-0.19)`에서
로봇 옆 `(12.63,0.36)`으로 점프 → 방향 오차 > 0.95 rad → tracker가 `v=0`으로 제자리 회전(`w=+0.55`) → 1초 뒤 경로가 반대쪽으로 뒤집혀 반대로 회전.

## 6. 코드 변경 요약 (커밋 `7488272`, `3d4c1e0`)

- **적응형 탈출 반경:** 정상 반경(0.44m)으로 못 찾으면, 로봇 자신이 비충돌인 가장 큰 반경(0.33 하한 ~ 0.36 상한)으로 재탐색. 지금보다 가까워지지 않는 경로만 통과. 로그에 `tier=escape`, 사용 반경 기록.
- **곡률 샘플링 시작 방향:** `arc_heading_offsets_rad = [0, ±0.4, ±0.8]` (tracker 제자리 회전 기준 아래).
- **제자리 회전 보호:** 장애물이 회전 반경 0.38m 안이면 recovery sweep을 내지 않음(`recovery_sweep_blocked`).
- **진단:** `clearance_probe`(반경 0.30/0.34/0.38/0.44/0.50m 중 어디까지 지도에 닿는지)를 거부/실패 이벤트에 기록.
- **tracker:** 통로 폭 0.42 → 0.34m, 비상 정지 0.38 → 0.30m. `minimum_turn_speed 0.08`, `rotate_in_place_angle 0.95 → 1.35`(설정값).
- **후보 선택 연속성:** `candidate_select.py` — 직전 경로와의 거리, 초기 방향 편차(한계 0.9 rad)로 선택. 기본 가중치 0(기존 동작), 설정에서 활성화.
- 안전 여유를 줄이는 변경이 포함되어 있다(통로 폭, 비상 정지선, 탈출 반경). **e-stop 담당자 상주 필수.**

현장 효과: 7488272는 B, C에서 확인. **3d4c1e0(연속성/회전 기준)의 효과는 미확인**(D가 AMCL 수렴 문제로 교란됨).

## 7. 지도 해상도 시험 (적용하지 않음)

롤링 지도 해상도 0.10 → 0.05m는 충돌 검사가 약 2.7배(39 → 106µs), 회피 호 330개 전수 검증이 0.32 → 1.21초로 계획 주기(0.8초)를 넘고,
통과 가능한 호 수도 거의 같았다(15 → 12). 참고: `point_occupied_with_radius`가 반경에 `0.5×해상도`를 더하므로 0.10m 해상도의 실효 반경은 약 +0.05m.

## 8. 오늘의 실수와 정정

- 부하 수치를 평균값(`pcpu`)으로 잘못 판단함(위 3절).
- 정지 상태 전용 측정기/감시기의 오탐: odom 값이 비면 "발산"으로 판단해 arm 직후 자동 disarm, 출력을 `head`로 잘라 **주행 중 감시 스크립트가 끊김**
  (바로 재부착, 주행 영향 없음). 감시 스크립트는 파일로만 출력하도록 수정, 값이 비면 판단하지 않도록 수정.
- arm 전 AMCL 수렴 확인을 조건으로 넣지 못함(D의 원인 중 하나).

## 9. 모델 vs 보조 규칙 보고에 대해

피드백(모델이 직접 장애물을 회피하는지 확인)에 대해 합의한 방향: (1) fallback과 recovery를 끈 **모델 단독 시험**, (2) RViz에 모델 후보 경로와
실제 실행 경로, **실행 주체**(MODEL / FALLBACK) 표시, (3) 주행별 모델 주행 비율 집계(예: 09-30 82%, 10-06 B 90%).
실행 주체를 숨기거나 모델 주행처럼 보이게 하는 표시는 만들지 않는다. 구현은 아직 하지 않았다.

## 10. TODO

**다음 시험 (순서)**
- [ ] arm 전 **AMCL 수렴 절차 고정:** pose 클릭 → RC로 1~2m 이동 → 표준편차 x ≤ 0.15m 확인 → goal → arm. 감시 스크립트의 AMCL 게이트를 이 기준으로 강화.
- [ ] 장애물을 한쪽으로 옮겨 반대편 틈 **1m 이상**에서 반복 시험(연속 3회). 틈 실측값 기록.
- [ ] `3d4c1e0`(연속성/회전 기준) 효과 확인: C와 같은 22m 직선 재주행, 멈춤/회전 구간 수 비교(rosbag + `tracker_diagnostics`).
- [ ] 복도 가운데 장애물(틈 0.7~0.9m) 처리 방침 결정. 하한(0.33→0.30m)을 낮추는 것은 접촉 위험으로 비권장.
- [ ] 코너/방향 전환 경로 시험(35m goal은 코너 포함 가능성). stability의 L자 경로 검증(`6ecc8f3`) 이식 검토.
- [ ] **보행자 시험:** dry-run 검출 확인 → 넓은 곳 → 복도(사람 1.20m 예측 거리 조건 때문에 복도는 "정지 후 대기"가 될 가능성).

**구현**
- [ ] 모델 단독 모드 스위치(`recovery sweep`, 옆 우회 끄기), RViz 실행 주체 표시, 주행별 모델/보조 비율 자동 집계.
- [ ] 모델 후보 순위가 매 사이클 바뀌는 문제(1→4→1→3→2…) 완화: 연속성 가중치 튜닝, 필요 시 stability의 selection continuity 이식.
- [ ] `fast_livo` CPU/메모리 증가(주행 거리 비례) 완화: 지도 윈도(`half_map_size`) 축소 시험 또는 긴 주행 전 재시작 절차화.

**배포/정리**
- [ ] 수정된 FAST-LIVO 빌드, 벡터화 `filterpass`, `fast_livo_image_enable:=false`, drive/ped 모드 스크립트를 **이미지/launch 기본값에 굽기**. IMU 단위 수정(`stability-20260921-continued`) 포함 여부 결정 후 비교 시험.
- [ ] stability 브랜치에 fallback/탈출 반경/연속성 이식 후 stability 기반 이미지 빌드.
- [ ] 디스크 사용률 97%(여유 34GB): 쓰지 않는 이미지(22GB급 4개) 정리.
- [ ] 정리: `xhost -SI:localuser:root`, `capstone_navi`의 깨진 ref `refs/codex/turn-diffs/...`, `jackal-backup-*` 폴더. 재부팅하면 네트워크 별칭(`configure_jackal_network.bash up`)과 호스트 `/tmp`(스크립트 사본)이 사라지므로 시작 스크립트는 이미지(`/root/*.bash`) 기준으로 관리.

## 11. 재현 메모

- 재부팅 후: `configure_jackal_network.bash up`(sudo) → `xhost +SI:localuser:root` → 컨테이너 기동(`run_stability_container.bash`, `IMAGE=moai-jackal-spubert:fallback-1006`).
- 스택: `/root/start_drivers.bash`(Livox, 필요 시 RealSense) → `/root/start_lio_fixed.bash`(drive 모드) → `/root/start_nav2_bridge.bash` → `/root/start_bag3.bash` → `/root/monitor_run.bash`(AMCL 게이트 + arm + 감시).
- 감시 스크립트 출력은 `head` 등으로 자르지 말 것(SIGPIPE로 감시 종료).
- 로그: 호스트 `~/jackal_logs/pedtest_20261006_103322/` (rosbag `bag_104853`, `bag_111007`, `bag_112317`, `bag_112804`, 약 500MB).
- push: `gh auth switch --user wnsdnnn` 후 push, 끝나면 `minsong0206`으로 복귀.
