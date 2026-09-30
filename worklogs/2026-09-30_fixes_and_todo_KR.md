# 2026-09-30 작업 로그: 정적 장애물 회피 주행 성공, 원인 분석, TODO

앞선 문서 `2026-09-30_fallback_field_test_KR.md`(오전 절차 정리)의 후속이다. 이 문서는 하루 동안의 실제 결과와
정정 사항을 기준으로 작성했으며, 앞 문서와 다른 내용은 이 문서가 우선한다.

## 1. 결과 요약

- **정적 장애물(복도 왼쪽 박스, 통로 폭 약 1.1m)을 지나 goal까지 멈춤 없이 주행에 성공했다.**
  약 5.8m를 18초(0.25 m/s)에 이동했고 goal 도달 후 자동 disarm(`goal_reached_disarmed`)됐다.
- 동적 보행자 회피는 **실제 로봇에서 아직 검증하지 못했다.** 모든 주행에서 `human=inf`(미검출)였다.
- 코드는 `capstone_navi`(브랜치 `jackal-real-drive-fixes-20260912`)에 커밋 `4d30059`로 push했다. 오프라인 테스트 130개 통과.

## 2. 오늘 찾은 근본 원인

| # | 증상 | 원인 | 조치 |
|---|---|---|---|
| 1 | 로컬라이제이션이 깨지고 odom이 수만 m로 발산, 이후 프로세스 종료 | **FAST-LIVO가 VIO 켜진 채 실행돼 메모리가 초당 20~25MB 증가**, 25GB에서 호스트 OOM killer가 종료(2회 확인, `dmesg`). README 터미널 3 명령에 `fast_livo_image_enable:=false`가 빠져 있었다. | 옵션 추가. RSS 150~230MB로 안정. README 수정 반영. |
| 2 | 장애물 앞에서 반복 정지 (브릿지는 유효 경로를 내는 중) | `safe_path_tracker`의 정지 규칙: 전방 ±40° 부채꼴에 0.75m 이내 점이 있으면 **경로와 무관하게 정지**. 박스 모서리가 그 안에 계속 남음. | 몸이 쓸고 지나가는 직진 통로(±0.42m) 기준 정지로 변경. |
| 3 | 모델 후보가 전부 거부되면 정지 | 후보 전부 탈락 시 대안 없음 | 아래 fallback 체인 추가. |
| 4 | 좁은 통로(안전 폭 0.1~0.2m)를 못 찾음 | 기존 옆 우회 오프셋(≥0.6m)이 너무 큼 | 경로를 ±0.30m까지 0.03m 간격으로 평행이동해 탐색 + 곡률 샘플링 |

## 3. 코드 변경 (`capstone_navi` 커밋 `4d30059`)

- `safe_path_tracker_node.py`, `navigation_core.py`: 통로 기반 정지 규칙(`obstacle_corridor_half_width > 0`일 때만 동작). 설정값
  `obstacle_stop_distance 0.45`, `obstacle_slow_distance 1.0`, `obstacle_corridor_half_width 0.42`, `obstacle_hard_stop_distance 0.38`.
  통로 밖 옆 장애물은 정지시키지 않고, 통로 안 장애물은 정지하며, 부채꼴 안 0.38m 이내는 기존처럼 비상 정지한다.
- `real_jackal_spubert_bridge_node.py`: 후보 전부 거부/guidance 실패 시 fallback 체인
  1) Nav2 경로 앞부분을 옆으로 평행이동한 안전 경로(`route_fallback.py`)
  2) 모델을 쓰지 않는 등곡률 호 경로, 시작 방향 ±0.4/0.8/1.2 rad 포함(`arc_planner.py`)
  3) 기존 blocked hold / recovery sweep.
  모든 fallback 경로는 모델 후보와 동일한 `validate_candidate_path`(footprint 충돌, 보행자 ≥1.20m 예측 거리, 진행도)를 통과해야 쓴다.
  한 에피소드는 60초로 제한한다. 사용/거부 이유는 `global_path_fallback`, `global_path_fallback_rejected`, `lateral_detour_failed` 이벤트로 기록된다.
- `guided_spubert_runtime.py`: `lateral_guidance_point` 오프셋에 0.3, 0.45 추가.
- `config/real_jackal_social005.yaml`: `static_safety_margin` 0.16 → 0.10 (footprint 0.44m). 이 검사는 AMCL이 아니라 라이다 롤링 맵 위에서 수행한다.
- `README_KR.md`: FAST-LIVO 명령에 `fast_livo_image_enable:=false`와 이유를 명시.
- 테스트 34개 추가(총 130개).

## 4. 주행 기록 요약 (마지막 주행)

- 예측 250번 중 206번 유효(82%). fallback이 이어서 진행시킨 횟수 44번(경로 앞부분 40, 곡률 샘플링 4).
- 경로 평행이동 사용 값: 오른쪽 0.21~0.27m. 곡률 샘플링 선택: 시작 방향 −1.2/−0.4/−0.8 rad 호.
- 정지성 hold는 `adaptive_guidance_failed` 8번 등 짧게 지나갔고 tracker는 계속 `tracking v=0.25`.
- 로그 위치(호스트): `~/jackal_logs/fallback_test_20260930_102617/` (rosbag `bag2_104711`, `monitor_bag`, 각 프로세스 로그, `fastlivo_rss.log`, `amcl_params_before.txt`). 진단 jsonl은 `~/jackal_logs/spubert_real_jackal_diagnostics.jsonl`.

## 5. 정정 사항 (오늘 스스로 틀렸던 것)

- **보행자가 로컬라이제이션을 깬다는 추정과 AMCL 파라미터 조정은 잘못된 층이었다.** 실제 원인은 위 1번(FAST-LIVO 메모리/발산)이었다. AMCL 파라미터는 원래 값으로 되돌렸다.
  참고로 `z_rand`는 보행자 같은 이상 빔에 대한 허용도라 낮추면 오히려 취약해진다(올려야 한다).
- 정지 상태 전용으로 만든 `odom_watch.py`(5초 중앙값 대비 10cm 이상 이탈)를 주행 중에 사용해 **실제 이동을 odom 튐으로 오탐하고 자동 disarm**시켰다. 주행 중에는 사용하면 안 된다.
- memory/이전 기록의 "`stability-20260921-continued` 이미지에 수정 5개가 모두 들어 있다"는 **FAST-LIVO에 대해 틀렸다.** 세 이미지(`collision-diag-v1`, `stability-20260921-continued`, `fallback-1e651b2`)의
  `fastlivo_mapping` 바이너리가 동일한 8/24 빌드이고 소스에도 img_buffer 상한(`kMaxImgBufferSize`)이 없다. 수정 소스는 별도 저장소 `spu_deploy_docker`(`ed0512c`, `fca86c0`, `30349ff`)에만 있다.
- `stability_0916` 저장소에는 FAST-LIVO 수정이 없다(네비게이션 코드 저장소).

## 6. 이 컨테이너에만 있는 수정 (이미지에 굽기 전)

- 컨테이너 `moai_jackal_spubert_fallback` (`--rm`이라 멈추면 사라짐). **오늘 상태를 이미지 `moai-jackal-spubert:fallback-0930-fixed`(22.6GB)로 보존했다.**
- `/root/fastlivo_fix_ws`: `spu_deploy_docker`의 수정된 FAST-LIVO를 별도로 빌드한 워크스페이스(약 1.5분). 사용하려면 `source /root/fastlivo_fix_ws/install/setup.bash`를 robot_ws 다음에 소싱한다.
- `/root/robot_ws/src/mola_bringup/{scripts,mola_bringup}/filterpass.py`: 벡터화 버전으로 교체(원본 백업 `/root/backup_before_0930_deploy_fixes/`).
- 컨테이너 안 코드 사본: `/root/moai_stability_ws/src/moai_jackal_spubert`(저장소 커밋 `4d30059`와 해시 동일 확인).

## 7. 재현 절차 (오늘 사용한 순서)

1. 호스트 네트워크: `bash real_jackal/scripts/configure_jackal_network.bash up` (sudo 필요, 서브커맨드는 `up`/`check`뿐이며 재부팅·케이블 분리 시 주소가 사라진다).
2. 컨테이너: `CAPSTONE_CALIB=~/capstone_navi/spu_deploy_docker/context/calib IMAGE=moai-jackal-spubert:fallback-0930-fixed CONTAINER_NAME=moai_jackal_spubert_fallback MOAI_ENABLE_GUI=1 ./real_jackal/docker/run_stability_container.bash`
3. 센서: RealSense, Livox 드라이버, `full_stack.launch.py ... fast_livo_image_enable:=false ...`, `filterpass.py`, `pointcloud_to_laserscan` (README 5절). 필요하면 위 FAST-LIVO 오버레이를 소싱한다.
4. Nav2: `nav2_route_planner.launch.py map:=/root/jackal_maps/world_map.yaml`, RViz `2D Pose Estimate` 후 로봇을 조금 움직여 수렴.
5. RViz: `xhost +SI:localuser:root`, `LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe`, `Frame Rate: 3` 복사본 사용.
6. 브릿지/tracker: `real_jackal_spubert.launch.py cmd_vel_topic:=/j100_0519/cmd_vel ...` → `verify_jackal_link.bash preflight` / `ready` → goal 지정 → e-stop 담당자 대기 → `enable_motion true`.
7. 종료: `enable_motion false`, launch는 프로세스 그룹 단위로 SIGINT 후 잔존 프로세스 확인(자식 노드가 남는 문제 있음), 마지막으로 남는 것은 강제 종료.

## 8. TODO

**다음 테스트 (우선)**
- [ ] 동적 보행자 테스트: (1) dry-run으로 검출 확인(`/ped_tracking`, 브릿지 `human_min`), (2) 넓은 곳에서 정지한 사람 → 걸어오는 사람, (3) 복도. 항상 e-stop 담당자 대기.
  복도(폭 약 2.2m)에서는 보행자 예측 거리 1.20m 조건 때문에 "회피"가 아니라 "정지 후 대기"가 될 가능성이 높다. `minimum_human_center_distance`, `human_stop_distance` 조정 여부를 결정한다. 사람 근처에서 3초 후 시작되는 recovery sweep(제자리 회전)의 안전성도 검토한다.
- [ ] 주행 중 로컬라이제이션 안정성 재검증(FAST-LIVO를 `image_enable:=false`로 둔 상태에서). 사용자 가설: 주행 시 입력 정보가 늘어나며 깨진다. 주행 중 odom/`cmd_vel`/AMCL 표준편차/`map→odom`/보행자 수/CPU를 함께 기록하고, 이동 중에도 쓸 수 있는 감지기로 교체한다.
- [ ] AMCL 초기 수렴 개선(재시작 후 표준편차 x≈0.5m, yaw≈14°에서 시작).

**배포/정리**
- [ ] 수정된 FAST-LIVO 빌드, 벡터화 `filterpass`, `fast_livo_image_enable:=false`를 **이미지/launch 기본값에 굽기**(컨테이너마다 다시 하지 않도록). `stability-20260921-continued`에도 이 수정이 없음.
- [ ] 디스크 사용률 96%(여유 약 40GB). 쓰지 않는 이미지(22GB급 여러 개) 정리.
- [ ] 네트워크 `up`, RViz 환경, `xhost` 절차를 README/스크립트에 반영. 오늘 `xhost +SI:localuser:root`를 추가했으니 필요 없어지면 `xhost -SI:localuser:root`로 되돌린다.

**코드 통합**
- [ ] 검증된 fallback(`route_fallback.py`, `arc_planner.py`, tracker 통로 규칙, 브릿지 helper 2곳)을 **stability 브랜치(`jackal-stability-20260916`)에 이식**하고 테스트. stability에는 route 변환 캐싱, MGP 원본 샘플 보존, plan_context, phase timing 등 현장 검증된 수정이 있고 `capstone_navi`에는 없다. 반대 방향 병합은 브릿지·tracker에서 20군데 이상 충돌한다.
- [ ] stability 코드 기반 이미지를 새로 빌드해 현장 테스트.

**조사**
- [ ] FAST-LIVO가 `image_enable:=false`에서도 완만히 증가(약 7MB/min, CPU 54%→102%/15분): 원인 확인. 당분간 테스트 사이에 스택 재시작.
- [ ] fallback 사용 패턴 오프라인 분석(사용 빈도, 평행이동량, 곡률 분포). 필요하면 파라미터 튜닝.
- [ ] `odom_watch.py` 폐기 또는 이동 인식 버전으로 교체.
- [ ] `capstone_navi` 저장소의 깨진 ref `refs/codex/turn-diffs/...` 삭제(원격 `fetch` 실패 원인), 사용하지 않는 `jackal-backup-*` 폴더 정리.

## 9. 참고 (Git)
- `y-u-n-j-u/moai_social_bert_refactored`에 push하려면 `gh auth switch --user wnsdnnn` 후 push, 끝나면 `minsong0206`으로 되돌린다(기본 계정은 403).
