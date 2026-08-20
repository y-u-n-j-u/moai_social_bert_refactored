# Intersection Safe-Teacher Pilot

## 목적

기존 `agents_training_intersection_low.yaml`은 비교 기준으로 보존한다. 새
`agents_training_intersection_safe_teacher_low.yaml`은 로봇이 교차로 바깥에서
출발하고 한 명의 보행자가 수직으로 횡단하는 V2 수집 시나리오다.

V2 pilot은 다음 조건을 사용한다.

- 보행자는 첫 Nav2 goal과 동시에 출발
- one-way coupling: 보행자는 계획된 경로를 유지하고 Nav2 teacher가 양보
- 사람 obstacle cloud 10 Hz
- 현재/예측 위치 ring 16점
- 사람 반경에 0.65 m safety margin 추가
- constant-velocity prediction 2.4초
- teacher 전용 Nav2 설정에서 MPPI 후진 금지(`vx_min=0.0`)
- 로봇은 `(6, 0)`에서 출발해 `(-6, 0)`으로 이동

## 10-seed 실행

기본 10회는 로봇 시작 x와 횡단 y를 서로 다르게 둔 10개 deterministic encounter profile을 사용한다. 좁은 수평 corridor의 footprint 여유를 보장하기 위해 목표 y는 중심선 기준 -0.2~+0.2m로 제한한다. `HUNAV_AUTO_GOAL_SEED`만 바꾸면 수동 HuNav 시나리오는 달라지지 않으므로, 품질 게이트는 seed뿐 아니라 실제 시작점-목표 profile의 고유성도 검사한다.

```bash
cd /home/junwoo/capstone/moai_social_bert_refactored/gazebo_classic
./run-safe-teacher-pilot.bash
```

빠른 단일 seed 검증:

```bash
HUNAV_PILOT_NAME=intersection_safe_teacher_smoke \
HUNAV_PILOT_SEEDS='101' \
./run-safe-teacher-pilot.bash
```

결과는 다음 경로에 seed별로 분리된다.

```text
hunav_gz_classic_ws/moai_recordings/<pilot-name>/
  seed_101/
    seed_101_raw.pkl
    processed/
    qualification.json
    runner.log
  pilot_summary.json
```

## 승인 기준

`teacher_quality_gate.py`는 window만 골라내지 않고 recording 전체를 판정한다.
다음 조건 중 하나라도 실패하면 해당 seed 전체를 positive 학습 데이터로 승인하지
않는다.

- 기록된 모든 window에서 사람 중심거리 1.2 m 이상
- 물리 충돌 기준 0.65 m 미만 window 0개
- map collision, 속도, 가속도, yaw-rate, timing 위반 0개
- route guidance 실패 0개
- `clean_social` window 1개 이상
- goal 동기화와 one-way coupling metadata 일치

10개 seed가 모두 통과하기 전에는 이 설정으로 대량 수집하지 않는다. 실패한 raw
recording은 hard-negative/evaluation으로 보존한다.

## 승인된 V3 결과

`intersection_safe_teacher_v3_corridor_safe_20260810`에서 고유 profile 10개가
모두 통과했다.

- raw window: 63
- clean social window: 36
- 전체 최소 사람 중심거리: 1.332 m
- 사람거리, map, route, kinematic, timing 위반: 0
- 결과: `hunav_gz_classic_ws/moai_recordings/intersection_safe_teacher_v3_corridor_safe_20260810/pilot_summary.json`

이 설정은 단일 보행자 횡단형 low-density 수집에만 승인한다. 기존 baseline과
실패한 y=+0.4 기록은 삭제하지 않고 비교 및 hard-negative 평가에 사용한다.
