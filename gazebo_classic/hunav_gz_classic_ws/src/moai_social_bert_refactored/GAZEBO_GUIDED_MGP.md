# Gazebo Guidance-conditioned MGP

이 브랜치는 Gazebo robot + pedestrian + local-map PKL을 윤주 SPU-BERT에
직접 넣고, 교수님이 제안한 순서로 goal을 선택하도록 통합한 버전이다.

```text
Gazebo PKL
  -> target-frame trajectory / pedestrian / guidance / map adapter
  -> MGP가 t+12 후보 goal 20개 생성
  -> free(1) 후보만 유지
  -> guidance point에 가장 가까운 후보 1개 선택
  -> 선택 goal을 TGP에 넣어 12-step trajectory 1개 생성
```

## 입력 계약

`goal_lbl`과 `guidance_lbl`은 같은 값이 아니다.

- `traj_lbl (12,2)`: target의 실제 미래 12개 좌표
- `goal_lbl (2,)`: `traj_lbl[-1]`, 즉 실제 t+12 도달점
- `guidance_lbl (2,)`: RViz final goal 방향 최대 8 m guidance point

MGP와 TGP는 모두 `(58,2)`를 유지한다.

```text
MGP: 0 SOT | 1:9 obs | 9:20 PAD×11 | 20 MASK | 21 GUIDANCE | 22:58 pedestrians
TGP: 0 SOT | 1:9 obs | 9:21 MASK×12 | 21 selected/label goal | 22:58 pedestrians
```

map은 로봇 진행방향이 `+x`가 되도록 회전한 `(32,32)` grid다. 값은
`0=unknown/padding`, `1=free`, `2=occupied`이며, Gazebo의 row 방향을
윤주 lookup 규약에 맞추기 위해 adapter에서 y축을 한 번 뒤집는다.
`16x16` patch 네 개를 펼쳐 `env_spatial_ids (4,256)`으로 입력한다.

후보 선택기는 NaN/Inf, map 밖, unknown, occupied 후보를 제거한다. 모든
후보가 제거되면 `selected_goal_valid=False`, index `-1`, goal `[0,0]`을
반환한다. 실제 robot runtime은 이 flag에서 정지하거나 재계획해야 한다.
TGP가 만든 12개 좌표도 같은 map 규칙으로 전부 검사하며,
`execution_valid = selected_goal_valid AND trajectory_map_safe`를 함께
반환하고 false이면 trajectory를 0으로 바꿔 정지시킨다. 이것은 point 기준
1차 검사이므로 실제 주행 전에는 robot footprint, inflation, 동적 장애물을
반영하는 Nav2 검증도 필요하다.

## 중요한 코드

- Dataset/map adapter:
  `SPU-BERT/spubert/datasets/moai_social_nav_extended_goal.py`
- MGP pool index, 후보 필터, guided inference:
  `SPU-BERT/spubert/model.py`
- guided/legacy GoalPooler와 scene encoder:
  `src/model.py`
- guided test/evaluation:
  `SPU-BERT/spubert/training.py`
- config 등록과 안전성 검사:
  `configs/loader.py`
- dataset loader 연결:
  `src/data_loader.py`
- map collision metric:
  `src/loss.py`
- empty-cluster-safe K-means:
  `src/utils.py`
- training loop와 token-contract 보고:
  `src/trainer.py`
- full training YAML:
  `configs/spubert/moai_social_nav_ext_scene_guided_fs.yaml`

adapter 하나만 복사하면 안 된다. 위 모델·trainer·config 변경이 함께 있어야
교수님이 요청한 guided MGP 경로가 실행된다.

## 검증

수집한 Gazebo split에 대해 아래 한 명령이
단위 테스트, 전 sample 입력 검증, RTX GPU forward/backward, guided 및
기존 multi-candidate inference를 순서대로 실행한다.

```bash
cd /path/to/cloned/repository
./scripts/validate.sh
```

확인하는 핵심 output shape은 다음과 같다.

```text
MGP candidates       (B,20,2)
guided selected goal (B,2)
guided trajectory    (B,12,2)
multi trajectories   (B,20,12,2)
```

2026-07-29 최종 확인 결과:

- Docker image: `moai-social-bert:cu124`
- unit test: 9/9 통과
- Gazebo input: 325/325 전수 통과
- RTX 4090, batch 32: forward/backward + guided/multi inference 통과
- batch 32 최대 allocated GPU memory: 약 670 MiB
- 1 epoch train → val → guided test → `model_best.pth` 저장 및 재로드 통과

1 epoch 수치는 코드 경로를 확인하는 smoke 결과일 뿐, 학습된 모델의 최종
성능으로 해석하면 안 된다.

실제 test sample의 map, MGP 후보, 선택 goal, TGP trajectory와 STOP 사례를
발표용 PNG로 다시 만들려면:

```bash
cd gazebo_classic/hunav_gz_classic_ws/src/moai_social_bert_refactored
./scripts/run_gazebo_visualization_docker.sh
```

기본 출력 위치는 모델 프로젝트의
`figures/gazebo_guided_mgp_results/`다.

## 학습

윤주 checkpoint와 원 실행환경을 받지 못했으므로 현재 YAML은
`train_mode: fs`인 from-scratch 재현 설정이다. 학습을 시작하려면:

```bash
cd /path/to/cloned/repository
./scripts/train.sh
```

먼저 설정만 확인하려면:

```bash
./scripts/train.sh --dry_run
```

윤주 checkpoint를 나중에 받으면 해당 checkpoint가 기대하는 원 config를
확인한 뒤 `train_mode`와 `runtime.pretrain_checkpoint`를 변경해야 한다.
guided 입력 의미가 기존 입력과 다르므로 checkpoint를 그대로 평가만 하지
말고 guided Gazebo 데이터로 fine-tuning해야 한다.

collision loss는 occupancy grid를 bilinear sampling해 예측 좌표로 gradient를
전달한다. 먼저 `col_weight: 0.0`을 기준으로 두고 동일한 route-GP 데이터에서
`0.1`, `1.0`을 비교한다. 서로 다른 weight의 total loss 값은 직접 비교하지
말고 held-out map collision rate와 closed-loop 주행 결과로 선택한다.
