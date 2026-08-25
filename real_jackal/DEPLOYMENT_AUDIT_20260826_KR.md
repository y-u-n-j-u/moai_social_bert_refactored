# 동료 Jackal Docker 코드 감사 결과

검사 대상:

- 별도 보관 폴더: `/home/junwoo/capstone/spu_deploy_docker_colleague`
- 원격 저장소: `git@github.com:y-u-n-j-u/spu_deploy_docker.git`
- 검사 커밋: `115824a48cc6e04814033f51bfa6646f40aa3de9`
- 검사일: 2026-08-26

## 확인된 구성

동료 저장소에는 다음 실차 구성요소가 실제 코드로 포함되어 있다.

- Livox MID-360 드라이버 설정과 FAST-LIVO2 odometry
- RealSense + YOLO 사람 탐지
- 2D 탐지와 LiDAR를 결합한 3D 보행자 위치 추정
- `/ped_tracking` ID 기반 보행자 추적
- `/scan` 기반 SPU-BERT 실시간 추론 노드
- Pure Pursuit 기반 Jackal 경로 추종 코드
- LiDAR/카메라 캘리브레이션 파일

체크포인트는 저장소에서 의도적으로 제외되어 있다. 원본 Dockerfile은
`context/model_assets/model_best.pth`와 `pretrain_model_best.pth`가 없으면 빌드되지
않는다.

## 그대로 사용하면 안 되는 부분

1. `realtime_predict_node.py`는 동료의 예전 ETH/UCY 체크포인트용 코드다.
   현재의 Adaptive GP, 경로 기반 GP 학습, MGP 안전 후보 Top-5, Social Loss 모델을
   사용하지 않는다.
2. 예전 노드는 미래 GT가 없을 때 후보 하나를 사실상 첫 번째 후보로 실행하는
   단순화가 남아 있다. 최신 런타임의 후보별 footprint/TGP/보행자 검사를 거치지 않는다.
3. `tracking_loop.py`에는 odometry timeout은 있지만 예측 경로 자체의 짧은 watchdog이
   없다. 추론 노드가 멈춘 뒤 오래된 경로를 계속 따라갈 가능성이 있다.
4. Docker README의 실차 연결은 아직 전체 하드웨어로 검증 완료된 상태가 아니다.
   특히 Jackal의 `ROS_DOMAIN_ID=1` 및 `/j100_0519/cmd_vel` 연결은 확인 진행 중이라고
   기록되어 있다.
5. `localize_world_map.launch.py`와 `planner_only.launch.py`에는
   `/home/yunju/...` 절대경로가 남아 있어 Docker의 `/root/...`에서 그대로 실행되지 않는다.
6. Dockerfile에 Nav2와 실제 야외 map 파일은 포함되어 있지 않다. 따라서
   `ComputePathToPose` 기반 경로 GP를 사용하려면 지도 제작, localization, planner가
   별도로 필요하다.
7. `full_stack.launch.py`가 base-to-LiDAR static TF를 이미 구성한다. 문서의 별도
   `static_transform_publisher`까지 동시에 실행하면 같은 TF를 중복 발행할 수 있다.
8. 동료 문서의 패키지명 `spu_bert_extractor`와 실제 실행 패키지
   `spu_data_extractor`, 예전 출력명 `/spu_bert/predict_path`와 실제 코드의
   `/spu_bert/predicted_path`가 혼재한다. 실제 코드를 기준으로 해야 한다.
9. LiDAR/카메라 extrinsic과 intrinsic은 센서 장착 위치가 바뀌면 다시 보정해야 한다.
10. `--privileged`와 `/dev` 전체 마운트는 초기 통합에는 편하지만 권한 범위가 넓다.

## 이번에 추가한 해결책

- Gazebo/HuNav 의존성이 없는 `moai_jackal_spubert` ROS 2 패키지 분리
- 최신 체크포인트 SHA256
  `03d88a96db9abe46c47d98837a57785f01c197a7338f0784510e88bec6a10a95` 고정
- Nav2 global path를 odom 좌표계로 변환한 뒤 Adaptive GP 생성
- 최신 MGP 후보와 Top-5 TGP의 지도, footprint, 진행성, 동적 보행자 거리 재검사
- `/scan`의 `+inf`를 열린 자유 공간으로 처리하는 rolling occupancy map
- 추론 실패, TF 실패, global path 부재, stale sensor에서 빈 경로를 발행하는 fail-closed 동작
- 경로/odom/scan/보행자 heartbeat watchdog과 전방 LiDAR emergency stop
- 제어 기본 출력 `/spu_bert/cmd_vel_dryrun`, 시작 시 항상 disarm
- 명시적인 `SetBool` 서비스로만 motion arm
- 동료 Docker를 base로 하고 최신 모델과 실차 패키지를 덮는 재현 가능한 overlay 이미지

## 아직 실제 장비에서 확인해야 하는 것

- 연구실 Jackal의 정확한 ROS domain, 네트워크 인터페이스, 실제 cmd_vel subscriber
- `/aft_mapped_to_init`의 header frame과 `odom -> base_link` TF 일치
- `/ped_tracking` 좌표가 현재 설정대로 `base_link` 상대 좌표인지
- 카메라와 LiDAR timestamp 지연 및 사람이 가려졌을 때 tracker heartbeat
- 실외 조명에서 YOLO 검출률과 LiDAR 재투영 오차
- 정지 상태와 저속에서 FAST-LIVO odometry 안정성
- 실제 Jackal footprint 및 제동거리에 맞춘 0.75m stop threshold
- 저장한 야외 지도에서 AMCL 수렴과 `map -> odom` TF

실차 성공을 주장하려면 위 항목을 monitor mode rosbag으로 먼저 기록하고, 저속 제어,
정적 장애물, 보행자 1명, 다중 보행자 순서로 단계적으로 통과해야 한다.
