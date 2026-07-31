# Gazebo 사회적 주행 데이터 수집부터 Guidance-conditioned SPU-BERT 학습까지

> 이 문서는 처음 프로젝트를 보는 사람이 **무엇을 만들었고, 데이터가 어떤 코드를 거쳐 어떤 모양으로 바뀌며, 현재 어디까지 검증되었는지** 한 번에 이해할 수 있도록 작성한 전체 가이드이다.
>
> 기준 저장소: `/home/kistmnl/social_nav/hunavsim_containers`  
> 기준 브랜치: `agent/hunavsim-gazebo-nav2-spubert-integration`

---

## 0. 한 문장 요약

**Gazebo + HuNavSim + Nav2에서 PMB2 로봇과 보행자의 주행 데이터를 수집하고, 이를 윤주 SPU-BERT의 trajectory/map token 입력으로 변환한 뒤, guidance-conditioned MGP가 여러 goal 후보를 만들고 지도상 유효한 후보를 골라 TGP가 미래 궤적을 예측하도록 학습·검증하는 파이프라인을 구축했다.**

전체 흐름은 다음과 같다.

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ 1. Gazebo / HuNavSim / Nav2                                                 │
│    PMB2 상태 + 보행자 상태 + Nav2 final goal + 정적 지도                     │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ logger
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 2. 원본 recording.pkl                                                       │
│    과거 8 step + 미래 12 step + final_goal + guidance_point + episode 정보   │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ 후처리 / 품질 필터 / local map 추출
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 3. clean_social.pkl                                                         │
│    target, neighbors, guidance, 20 m × 20 m local map                       │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ episode 단위 train / val / test 분할
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 4. 윤주 모델 Dataset adapter                                                │
│    로봇 기준 좌표 변환 → 이웃 4명 선택 → MGP/TGP token → map patch token     │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 5. Guidance-conditioned MGP → 지도 필터 → guidance 최근접 goal → TGP         │
│    여러 goal 후보 생성       장애물 제거       하나 선택       궤적 생성     │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ 6. 오프라인 평가                                                            │
│    ADE / FDE / GDE + goal 유효성 + trajectory map 안전성 + execution-valid  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. 먼저 구분해야 하는 세 가지

이 프로젝트를 이해할 때 가장 많이 혼동되는 부분이다.

### 1.1 데이터 수집 때 로봇을 움직이는 것은 Nav2이다

현재 Gazebo 데이터 수집 중 PMB2가 움직이는 이유는 **학습된 MGP/TGP가 로봇을 실시간 제어하기 때문이 아니다.**

- 자동 goal 노드가 목적지를 만든다.
- Nav2 global/local planner가 그 목적지까지 주행한다.
- logger는 그 주행 결과를 관찰하여 학습용 데이터로 저장한다.

즉, 현재 수집 데이터는 다음 구조의 **Nav2 demonstration 데이터**이다.

```text
자동 goal 생성 → Nav2 주행 → 로봇/보행자 궤적 기록
```

### 1.2 모델 학습은 수집이 끝난 뒤 별도로 수행한다

저장된 PKL을 정제하고 모델 입력으로 바꾼 다음, MGP와 TGP를 학습한다.

```text
recording.pkl → 후처리 → split → adapter → MGP/TGP 학습
```

### 1.3 현재 모델 평가는 오프라인 평가이다

현재 완료된 결과는 test 데이터의 과거 궤적과 지도를 모델에 넣고 미래 궤적을 예측한 뒤 정답과 비교한 것이다.

**아직 학습된 모델이 Gazebo 안에서 Nav2를 대체하거나, 매 제어 주기마다 goal/trajectory를 보내는 closed-loop 실험까지 완료한 것은 아니다.**

---

## 2. 디렉터리와 핵심 코드 지도

아래 경로는 저장소 루트 기준이다.

```text
hunavsim_containers/
├── scripts/
│   ├── run_simulation.sh                 # Gazebo 실행과 데이터 수집 진입점
│   ├── prepare_dataset.sh                # 후처리 + 전체 run 병합/split
│   └── train.sh                          # Docker 내부 모델 학습 실행
│
└── gazebo_classic/hunav_gz_classic_ws/
    ├── moai_recordings/                  # Gazebo 원본 PKL
    │
    └── src/
        ├── moai_hunav_bridge/
        │   ├── moai_hunav_bridge/
        │   │   ├── random_goal_publisher_node.py
        │   │   └── jackal_teleop_dataset_logger_node.py
        │   └── scripts/
        │       └── postprocess_pedestrian_dataset.py
        │
        └── moai_social_bert_refactored/
            ├── configs/spubert/
            │   ├── moai_social_nav_ext_scene_guided_fs.yaml
            │   ├── ethucy_all_scene_pretrain.yaml
            │   └── moai_social_nav_ext_scene_guided_ethucy_pt.yaml
            ├── scripts/
            │   ├── prepare_gazebo_splits.py
            │   ├── validate_gazebo_guided_dataset.py
            │   ├── prepare_ethucy_all_pretrain.py
            │   ├── visualize_training_log.py
            │   └── compare_guided_experiments.py
            ├── SPU-BERT/spubert/
            │   ├── datasets/moai_social_nav_extended_goal.py
            │   ├── model.py
            │   └── training.py
            ├── src/trainer.py
            ├── data/
            ├── output/
            └── figures/
```

---

## 3. 1단계: Gazebo에서 원본 데이터 수집

## 3.1 시뮬레이션 구성

수집 환경의 역할은 다음과 같다.

| 구성 요소 | 역할 |
|---|---|
| Gazebo Classic | 로봇, 사람, 장애물이 존재하는 물리 시뮬레이션 |
| HuNavSim | 보행자 생성과 사회적 행동 시뮬레이션 |
| PMB2 | 학습 대상이 되는 로봇 |
| Nav2 | 주어진 goal까지 PMB2를 이동시키는 navigation stack |
| RViz | 지도, 로봇, 경로를 확인하고 수동 goal을 줄 수 있는 UI |
| auto-goal node | 수동 클릭 없이 goal을 자동 생성 |
| dataset logger | 시간에 따른 로봇·보행자 상태를 window로 저장 |

## 3.2 실행 예

```bash
cd /home/kistmnl/social_nav/hunavsim_containers

HUNAV_AUTO_GOAL=True \
HUNAV_AUTO_GOAL_MAX_GOALS=0 \
./scripts/run_simulation.sh corridor_run_001
```

- `HUNAV_AUTO_GOAL=True`: 자동 goal 생성 기능 활성화
- `HUNAV_AUTO_GOAL_MAX_GOALS=0`: goal 개수 제한 없이 계속 실행
- `corridor_run_001`: 저장 파일을 구분하기 위한 recording 이름
- 종료할 때 `Ctrl+C`를 누르면 완료된 episode가 PKL에 저장된다.
- 종료 순간 아직 goal에 도착하지 못한 진행 중 episode는 버린다.

**중요:** 자동 goal을 무제한으로 만든다는 뜻이지, 한 goal을 향해 영원히 가는 뜻은 아니다. 하나에 도착하면 다음 goal을 만든다.

## 3.3 자동 goal 생성 코드

파일:

```text
gazebo_classic/hunav_gz_classic_ws/src/moai_hunav_bridge/
moai_hunav_bridge/random_goal_publisher_node.py
```

주요 흐름:

1. `_on_map()` — OccupancyGrid 지도를 받는다. (`146`행 부근)
2. `_on_robot()` — 현재 로봇 위치를 갱신한다. (`165`행 부근)
3. `_sample_goal()` — 이동 가능한 free cell 중 goal 후보를 뽑는다. (`327`행 부근)
4. `_validate_path()` — Nav2에 경로 계산을 요청하여 실제 도달 가능한지 확인한다. (`265`행 부근)
5. `_publish_goal()` — 검증된 goal을 Nav2에 발행한다. (`375`행 부근)
6. `_on_timer()` — 도착, timeout, 다음 goal 생성 상태를 관리한다. (`171`행 부근)

따라서 단순히 임의 좌표를 찍는 것이 아니라,

```text
free cell 후보 → Nav2 path 존재 여부 확인 → 유효한 goal만 발행
```

순서로 동작한다.

## 3.4 logger가 기록하는 것

파일:

```text
gazebo_classic/hunav_gz_classic_ws/src/moai_hunav_bridge/
moai_hunav_bridge/jackal_teleop_dataset_logger_node.py
```

> 파일명에는 `jackal`이 남아 있지만, 현재 통합 환경에서는 PMB2/HuNav 상태를 이용하는 데이터 logger 역할을 한다.

주요 콜백:

- `_on_robot()` (`97`행 부근): 현재 로봇 상태 수신
- `_on_humans()` (`101`행 부근): 현재 보행자 상태 수신
- `_on_goal()` (`108`행 부근): Nav2에 발행된 최종 목적지 수신
- `_append_sample_from_window()` (`217`행 부근): 과거/미래 window를 sample로 구성
- `_commit_pending_episode()` (`290`행 부근): goal에 정상 도착한 episode 확정
- `_discard_pending_episode()` (`298`행 부근): 실패/중단 episode 폐기
- `_flush()` (`327`행 부근): PKL 파일 저장

### final_goal은 어디서 오는가?

`final_goal`은 후처리 단계에서 임의로 만드는 값이 아니다. `_on_goal()`이 자동 goal node 또는 RViz/Nav2에서 발행된 goal 메시지를 받아 현재 episode에 보관한다.

### 각 trajectory window에 맞는 goal은 어떻게 찾는가?

goal 메시지가 들어오면 logger가 새 episode를 시작하고, 그 goal이 활성화되어 있는 동안 생성된 모든 sliding window에 같은 `episode_id`, `goal_stamp`, `final_goal`을 붙인다.

```text
goal A 수신
  ├─ window A-1 ─ final_goal A
  ├─ window A-2 ─ final_goal A
  └─ window A-3 ─ final_goal A
goal A 도착 → episode A commit

goal B 수신
  ├─ window B-1 ─ final_goal B
  └─ ...
```

그러므로 “sample 시간과 가장 가까운 goal을 나중에 검색”하는 방식이 아니라, **수집 순간 활성화된 episode의 goal을 직접 연결**한다.

## 3.5 관측 8 step과 미래 12 step

각 sample은 시간적으로 연속된 궤적을 window로 자른 것이다.

```text
시간 ───────────────────────────────────────────────▶

t-7  t-6  ...  t-1   t0 | t+1  t+2  ...  t+11  t+12
└────── 과거 8 step ────┘ └──── 미래 12 step ───────┘
                             학습 정답(label)
```

원본 sample의 대표 필드는 다음과 같다.

| 필드 | shape | 의미 |
|---|---:|---|
| `target_past` | `(8, 2)` | PMB2의 과거 x, y |
| `neighbor_past` | `(N, 8, 2)` | 주변 보행자 N명의 과거 x, y |
| `target_future` | `(12, 2)` | PMB2의 실제 미래 궤적 |
| `neighbor_future` | `(N, 12, 2)` | 보행자의 미래 궤적 |
| `final_goal` | `(2,)` | 해당 episode의 Nav2 최종 목적지 |
| `guidance_point` | `(2,)` | 현재 로봇에서 final goal 방향 최대 8 m 점 |
| `episode_id` | scalar/string | 같은 goal 주행을 묶는 식별자 |

---

## 4. 2단계: final goal에서 8 m guidance point 계산

파일:

```text
.../jackal_teleop_dataset_logger_node.py
```

함수:

```python
_guidance_point_from_goal(...)
```

(`310`행 부근)

현재 로봇 위치를 \(p=(x_r,y_r)\), 최종 목적지를 \(g=(x_g,y_g)\)라고 하면,

\[
d = g-p
\]

\[
L = \|d\|
\]

\[
guidance =
\begin{cases}
g, & L \le 8 \\
p + 8\frac{d}{L}, & L > 8
\end{cases}
\]

쉽게 말하면 다음과 같다.

```text
final goal이 8 m 안쪽: final goal 자체가 guidance point
final goal이 8 m 바깥: final goal 방향으로 정확히 8 m 앞의 점
```

```text
Robot ●────────────○────────────────────────★ Final goal
       최대 8 m     Guidance point
```

이는 기하학적으로 **현재 로봇을 중심으로 한 반지름 8 m 원**과 **현재 로봇에서 final goal로 향하는 직선**의 진행 방향 교점과 같다. 단, final goal이 8 m보다 가까우면 원의 교점까지 가지 않고 final goal을 사용한다.

후처리 코드에도 같은 fallback 계산이 있다.

```text
.../moai_hunav_bridge/scripts/postprocess_pedestrian_dataset.py
guidance_point_from_goal()                       # 381행 부근
```

원본 metadata에 guidance가 있으면 그대로 사용하고, 과거 파일처럼 guidance가 없을 때만 같은 규칙으로 다시 계산한다. (`420–423`행 부근)

---

## 5. 3단계: 원본 PKL 후처리와 학습용 sample 추출

파일:

```text
gazebo_classic/hunav_gz_classic_ws/src/moai_hunav_bridge/
scripts/postprocess_pedestrian_dataset.py
```

## 5.1 후처리가 필요한 이유

Gazebo가 기록한 모든 window가 좋은 학습 자료는 아니다.

예:

- 로봇이 거의 움직이지 않은 window
- 미래 이동량이 너무 작은 window
- 비정상적으로 빠른 순간 이동
- 보행자가 너무 멀어 사회적 상호작용이 없는 window
- 로봇과 사람이 이미 충돌에 가까운 비정상 sample
- 필요한 길이 8+12 step을 만족하지 못한 window

따라서 원본을 곧바로 학습하지 않고, 궤적 품질과 사회적 상호작용을 검사한다.

## 5.2 처리 순서

```text
원본 PKL
  │
  ├─ 궤적 길이와 NaN/Inf 검사
  ├─ 로봇 이동량/경로 길이/속도 검사
  ├─ 로봇-보행자 최소 거리 검사
  ├─ social interaction 범위 검사
  ├─ 지도에서 로봇 중심 local map 추출
  ├─ final_goal / guidance_point 연결
  └─ clean_all / clean_social / rejected로 분류
```

주요 함수:

| 함수 | 위치 | 역할 |
|---|---:|---|
| `load_map()` | 74행 | YAML과 PGM 정적 지도 읽기 |
| `world_to_pixel()` | 99행 | 세계 좌표를 지도 pixel로 변환 |
| `local_map_patch()` | 108행 | 로봇 중심 local map 생성 |
| `sample_quality()` | 266행 | 이동량, 속도, 거리 등 계산 |
| `pass_basic_filter()` | 361행 | 기본 품질 조건 검사 |
| `pass_social_filter()` | 373행 | 주변 사람과 유효한 상호작용 검사 |
| `make_model_sample()` | 401행 | 최종 학습용 dictionary 생성 |
| `main()` | 934행 | 파일 전체 처리 및 결과 저장 |

현재 기본 사회적 거리 기준은 다음 코드 인자에서 확인할 수 있다.

```text
--min-social-distance 0.7 m
--max-social-distance 3.0 m
--collision-distance  0.65 m
--guidance-radius      8.0 m
```

## 5.3 local map 추출

원본 정적 지도는 전역 지도이다. 모델은 매 sample마다 전체 건물을 볼 필요가 없으므로 현재 로봇 주변만 자른다.

```text
전역 지도
┌──────────────────────────────────────────┐
│                    █████                 │
│          ┌────────────────────┐          │
│          │  로봇 주변 20 m     │          │
│          │         R          │          │
│          │              ███   │          │
│          └────────────────────┘          │
└──────────────────────────────────────────┘
                       ↓
              32 × 32 local map
```

후처리 결과 `local_map`의 shape은 `(32, 32)`이다.

## 5.4 현재 사용한 데이터

유효한 7개 Gazebo recording에서 `clean_social` sample을 사용했다.

| recording | clean social samples |
|---|---:|
| corridor_low | 408 |
| corridor_medium | 223 |
| doorway_high | 9 |
| doorway_low | 69 |
| doorway_medium | 35 |
| intersection_low | 507 |
| intersection_medium | 242 |
| **합계** | **1,493** |

비어 있거나 정상 episode가 없는 recording은 학습 병합 대상에서 제외했다.

---

## 6. 4단계: 8 m 지도의 실패와 20 m 지도로 수정한 이유

이 과정은 현재 파이프라인에서 가장 중요한 수정 중 하나이다.

## 6.1 처음 설정의 문제

초기 local map은 전체 폭이 8 m였다.

```text
8 m × 8 m local map = 로봇 기준 x,y 각각 약 -4 m ~ +4 m
```

그런데 guidance point는 최대 8 m 앞에 있을 수 있다. 미래 12 step의 endpoint도 4 m 범위를 벗어날 수 있다.

```text
8 m local map 범위
          +4 m
     ┌─────────┐
     │    R ───┼────── G  (guidance 최대 8 m)
     └─────────┘
          -4 m
```

이 경우 정상적인 goal조차 지도 바깥으로 분류된다. 실제로 초기 지도 설정에서 `execution-valid` 비율이 약 **6.7%**에 불과했다.

이는 모델이 나빠서가 아니라 **모델이 판단에 사용하는 지도 범위가 목표 범위보다 작았기 때문**이다.

## 6.2 수정한 설정

다음과 같이 통일했다.

| 항목 | 값 |
|---|---:|
| local map 실제 크기 | `20 m × 20 m` |
| 로봇 중심 범위 | `-10 m ~ +10 m` |
| grid shape | `32 × 32` |
| 한 cell 해상도 | `20 / 32 = 0.625 m` |
| patch 크기 | `16 × 16 cells` |
| patch 수 | `2 × 2 = 4` |

관련 코드/설정:

```text
scripts/prepare_dataset.sh
  --map-size-m 20.0

configs/spubert/moai_social_nav_ext_scene_guided_fs.yaml
  local_map_size_m: 20.0
  local_map_grid_size: 32
  env_range: 10.0
  env_resol: 0.625
```

수정 뒤 전체 1,493 sample에서:

- guidance point 지도 범위 밖: **0**
- 정답 미래 endpoint 지도 범위 밖: **0**

**핵심:** guidance 최대 거리와 model map의 반쪽 범위가 서로 일치해야 한다. 현재는 guidance 8 m를 포함하기 위해 반쪽 범위를 10 m로 두었다.

## 6.3 현재 설정의 trade-off

20 m를 32 cell로 표현하므로 한 cell이 0.625 m이다. 범위는 충분하지만 작은 장애물을 세밀하게 표현하는 능력은 떨어질 수 있다.

향후 비교 실험 후보:

- `20 m / 32 × 32`: 현재 설정, 계산량이 작음
- `20 m / 64 × 64`: 더 정밀하지만 map token과 계산량 증가
- 같은 32 × 32에서 범위를 줄이기: 정밀하지만 8 m guidance가 경계에 가까워짐

---

## 7. 5단계: train / val / test 분할

파일:

```text
gazebo_classic/hunav_gz_classic_ws/src/moai_social_bert_refactored/
scripts/prepare_gazebo_splits.py
```

## 7.1 sample을 무작위로 나누면 안 되는 이유

sliding window는 서로 많이 겹친다.

```text
window 1: t0  ... t19
window 2:   t1 ... t20
window 3:     t2 ... t21
```

이들을 sample 단위로 무작위 분할하면 거의 같은 궤적이 train과 test에 동시에 들어간다. 그러면 test 성능이 실제보다 좋아 보인다.

## 7.2 episode-safe split

`group_samples()` (`64`행 부근)가 다음 두 값을 묶어 하나의 독립 그룹으로 만든다.

```text
(recording_id, episode_id)
```

한 goal로 주행한 모든 겹치는 window는 반드시 같은 split으로 이동한다.

```text
Episode A 전체 → TRAIN
Episode B 전체 → VAL
Episode C 전체 → TEST
```

`validate_no_leakage()` (`155`행 부근)가 split 사이 group 교집합이 0인지 검사한다.

## 7.3 현재 분할 결과

| split | samples |
|---|---:|
| train | 1,044 |
| validation | 224 |
| test | 225 |
| **합계** | **1,493** |

- 전체 독립 episode: **122**
- train/val/test episode 중복: **0**
- split seed: **20**

---

## 8. 6단계: Gazebo dictionary를 윤주 모델 입력으로 변환

이 역할을 하는 코드가 흔히 말한 **adapter**이다.

파일:

```text
gazebo_classic/hunav_gz_classic_ws/src/moai_social_bert_refactored/
SPU-BERT/spubert/datasets/moai_social_nav_extended_goal.py
```

클래스:

```python
class MoAISocialNavExtendedGoalDataset(Dataset)
```

(`28`행 부근)

### 왜 수집 단계에서 바로 모델 token으로 저장하지 않았는가?

수집과 모델 포맷을 분리하면 다음 장점이 있다.

- 모델의 이웃 수, token 규칙, map patch 크기가 바뀌어도 Gazebo를 다시 실행하지 않아도 된다.
- 원본 세계 좌표를 보존하므로 다른 모델에도 재사용할 수 있다.
- 전처리 오류를 수정해도 비싼 시뮬레이션 수집을 반복하지 않는다.
- 동일 원본에서 여러 전처리 설정을 공정하게 비교할 수 있다.

따라서 logger는 의미가 명확한 원본 값을 저장하고, adapter가 학습 직전에 모델 전용 tensor로 바꾸는 현재 구조가 적절하다.

## 8.1 전체 궤적 배열 구성

`build_all_trajs()` (`209`행 부근):

```text
target robot 1명 + neighbor N명
          ↓
하나의 trajectory 배열로 결합
```

타겟과 보행자를 같은 배열 형식으로 합치는 이유는 모든 agent에게 같은 좌표 변환, 거리 계산, 시야 필터를 한 번에 적용하기 위해서이다. 배열의 첫 번째 agent는 target robot이며 나머지는 neighbor이다.

## 8.2 로봇 중심 좌표계로 변환

`transform_to_target()` (`237`행 부근):

1. 과거 마지막 로봇 위치를 원점 `(0, 0)`으로 이동한다.
2. 로봇 진행 방향이 모델 좌표의 +x축을 향하도록 모든 궤적을 회전한다.

진행 방향은 마지막 두 과거 좌표로 구한다.

\[
v = p_t-p_{t-1}
\]

두 좌표를 빼면 이전 위치에서 현재 위치로 향하는 변위가 되므로 진행 방향 벡터가 된다.

```text
세계 좌표계                  모델의 로봇 기준 좌표계

        p(t)                           +x
       ↗                               ─────────▶
 p(t-1)      장애물               R(0,0)      장애물
```

`transform_point()` (`251`행 부근)은 guidance/final goal 같은 단일 점에도 동일한 이동·회전을 적용한다.

**미래 좌표로 로봇의 현재 진행 방향을 계산하지 않는다. 과거의 마지막 두 점만 사용하므로 추론 시에도 사용할 수 있다.**

## 8.3 가까운 보행자 4명 선택

`neighbor_filtering()` (`255`행 부근):

- 로봇의 관측 범위/시야각 조건 적용
- 거리순 정렬
- 최대 `num_nbr=4`명 선택
- 부족한 neighbor 자리는 padding

모든 사람을 쓰지 않고 가까운 사람을 선택하는 이유는:

- 가까운 사람이 로봇의 즉각적인 회피 행동에 가장 큰 영향을 줌
- 입력 token 길이를 모든 sample에서 동일하게 유지
- 사람 수가 많은 장면에서도 계산량을 고정

## 8.4 MGP와 TGP token 만들기

`build_mgp_tgp_streams()` (`293`행 부근)가 핵심이다.

### MGP 입력

MGP는 12번째 미래 시점의 goal을 예측한다.

```text
[SOT]
+ target 과거 8개
+ 미래 1~11 step PAD 11개
+ 미래 12번째 위치 MASK 1개
+ guidance point 1개
+ neighbor 4명 × ([SEP] + 과거 8개)
```

길이 계산:

```text
1 + 8 + 11 + 1 + 1 + 4 × (1 + 8)
= 22 + 36
= 58 tokens
```

따라서:

```text
mgp_spatial_ids: (58, 2)
```

교수님이 말씀한 “MGP에 12개의 padding을 넣지 말고 11개의 padding과 마지막 mask를 넣는다”는 구조가 이 부분에 반영되어 있다.

### TGP 입력

TGP는 선택된 goal을 조건으로 12개 미래 위치를 복원한다.

```text
[SOT]
+ target 과거 8개
+ 미래 MASK 12개
+ goal token 1개
+ neighbor 4명 × ([SEP] + 과거 8개)
```

길이:

```text
1 + 8 + 12 + 1 + 4 × 9 = 58 tokens
```

따라서:

```text
tgp_spatial_ids: (58, 2)
```

## 8.5 최종 모델 입력 shape

한 sample을 `__getitem__()` (`94`행 부근)으로 읽으면 대표 tensor는 다음과 같다.

| key | shape | 의미 |
|---|---:|---|
| `mgp_spatial_ids` | `(58, 2)` | guidance가 포함된 MGP 좌표 token |
| `tgp_spatial_ids` | `(58, 2)` | TGP 좌표 token |
| `traj_lbl` | `(12, 2)` | 실제 로봇 미래 궤적 |
| `goal_lbl` | `(2,)` | 실제 12번째 미래 위치 |
| `guidance_lbl` | `(2,)` | 로봇 기준 guidance point |
| `env_spatial_ids` | `(4, 256)` | 4개 map patch의 cell 값 |
| `envs` | `(32, 32)` | 정렬·인코딩된 local map |
| `envs_params` | `(6,)` | map 좌표 해석용 파라미터 |

이 외에도 segment ID, temporal ID, attention mask가 함께 생성된다.

---

## 9. 7단계: local map을 윤주 모델의 map token으로 변환

같은 adapter 파일의 다음 함수들이 담당한다.

| 함수 | 위치 | 역할 |
|---|---:|---|
| `align_local_map_to_target()` | 366행 | 로봇 진행 방향 기준으로 지도 정렬 |
| `encode_local_map_for_colleague()` | 412행 | map 값 체계 변환 |
| `build_local_map_streams()` | 435행 | patch와 mask/id 생성 |

## 9.1 왜 map을 회전하는가?

trajectory는 로봇 진행 방향이 +x가 되도록 회전했는데 map을 그대로 두면 좌표계가 서로 맞지 않는다.

```text
trajectory: 로봇 앞 = 오른쪽(+x)
map:        로봇 앞 = 원래 세계 방향
```

이 상태에서는 모델이 “궤적 좌표의 장애물 위치”와 “map의 장애물 위치”를 연결할 수 없다. 따라서 궤적을 돌린 각도와 동일한 기준으로 map도 정렬한다.

쉽게 말하면 로봇에 카메라를 붙이고 항상 로봇이 화면 오른쪽을 보도록 지도를 돌리는 것이다.

## 9.2 역매핑이란 무엇인가?

회전 후 만들 최종 32×32 출력 grid의 각 cell을 하나씩 방문하면서 다음 질문을 한다.

> “이 출력 cell은 회전 전 원본 map의 어느 위치에서 온 것인가?”

그 원본 위치의 값을 읽어 출력 cell을 채운다.

```text
출력 cell (r, c)
       │ 역회전
       ▼
원본 map의 실수 좌표
       │ 보간/최근접 값 읽기
       ▼
출력 cell 값 기록
```

원본의 cell을 앞으로 회전시켜 출력에 찍는 forward mapping을 쓰면 여러 원본 cell이 같은 출력 cell에 겹치거나 아무 값도 도착하지 않는 구멍이 생긴다. 반대로 출력의 모든 cell에서 출발하는 inverse mapping은 각 출력 cell을 반드시 한 번 채울 수 있다.

회전했을 때 원본 범위를 벗어난 모서리는 unknown/padding으로 처리한다.

## 9.3 map 값 체계

Gazebo 후처리 단계의 local map:

| 값 | 의미 |
|---:|---|
| `0.0` | free |
| `0.5` | unknown 또는 crop 바깥 padding |
| `1.0` | occupied |

윤주 모델 입력으로 인코딩한 값:

| token 값 | 의미 |
|---:|---|
| `0` | unknown / padding |
| `1` | free |
| `2` | occupied |

이를 통해 unknown을 free로 잘못 해석하지 않도록 구분한다.

## 9.4 32×32를 4개의 patch로 나누기

```text
32 × 32 map

┌─────────────────┬─────────────────┐
│ Patch 0         │ Patch 1         │
│ 16 × 16         │ 16 × 16         │
├─────────────────┼─────────────────┤
│ Patch 2         │ Patch 3         │
│ 16 × 16         │ 16 × 16         │
└─────────────────┴─────────────────┘
```

각 16×16 patch를 한 줄로 펼치면 256개 cell이다.

```text
4 patches × 256 cells
→ env_spatial_ids shape = (4, 256)
```

각 patch에는 모델이 구분할 수 있도록 spatial 값뿐 아니라 segment ID, temporal ID, attention mask도 붙는다.

---

## 10. 8단계: Guidance-conditioned MGP와 TGP

모델 파일:

```text
gazebo_classic/hunav_gz_classic_ws/src/moai_social_bert_refactored/
SPU-BERT/spubert/model.py
```

## 10.1 MGP와 TGP의 역할

| 모듈 | 역할 |
|---|---|
| MGP | 12번째 미래 시점에 도달할 수 있는 여러 goal 후보 생성 |
| map filter | occupied/unknown/out-of-bounds 후보 제거 |
| guidance selector | 남은 후보 중 guidance point에 가장 가까운 하나 선택 |
| TGP | 선택된 goal에 도달하는 12-step 궤적 생성 |

```text
과거 궤적 + 사람 + 지도 + guidance
                 │
                 ▼
        MGP: goal 후보 20개
      ●  ●   ● █●  ●   ●
                 │
                 ▼
       지도 밖 / 장애물 후보 제거
      ●  ●      █    ●
                 │
                 ▼
       guidance와 가장 가까운 ● 선택
                 │
                 ▼
        TGP: 12-step trajectory
```

## 10.2 교수님 요구사항과 코드 대응

교수님 말씀의 핵심:

1. guidance point를 MGP query/condition으로 준다.
2. 미래 12번째 위치를 MASK로 두고 여러 goal을 예측한다.
3. occupied 영역에 생긴 goal은 없앤다.
4. 남은 후보 중 guidance point에 가까운 goal 하나를 고른다.
5. 그 goal에 도달하는 TGP를 생성한다.

구현 위치:

| 요구사항 | 코드 |
|---|---|
| guidance token을 MGP에 삽입 | dataset adapter `build_mgp_tgp_streams()` 293행 부근 |
| MGP 후보 생성 | `SBertPlusMGPModel.goal_predictor()` 723행 부근 |
| map cell 유효성 판정 | `classify_map_points()` 34행 부근 |
| guidance 최근접 유효 후보 선택 | `select_guided_goal_candidates()` 104행 부근 |
| 전체 MGP→filter→TGP 실행 | `SBertPlusFTModel.inference_guided()` 989행 부근 |
| TGP 추론 | `SBertPlusTGPModel.inference()` 605행 부근 |

`inference_guided()`의 코드 주석도 다음 흐름을 명시한다.

```text
MGP candidates -> map filter -> nearest guidance goal -> one TGP
```

## 10.3 모든 후보가 유효하지 않을 때

모든 MGP 후보가 occupied/unknown/out-of-map이면 정상적인 안전 후보를 선택할 수 없다. 이 경우 fallback 후보가 사용되며 평가에서는 `all_candidates_invalid`로 별도 집계한다.

따라서 단순 ADE/FDE뿐 아니라 “유효 후보가 하나라도 있었는가”도 반드시 함께 봐야 한다.

---

## 11. 9단계: 모델 구성과 학습 설정

Gazebo-only baseline 설정:

```text
.../configs/spubert/moai_social_nav_ext_scene_guided_fs.yaml
```

핵심 하이퍼파라미터:

| 항목 | 값 |
|---|---:|
| hidden size | 256 |
| encoder layers | 4 |
| attention heads | 4 |
| intermediate size | 1024 |
| dropout | 0.1 |
| map 입력 | 사용 |
| guidance-conditioned MGP | 사용 |
| MGP goal candidates | 20 |
| latent sampling `d_sample` | 400 |
| batch size | 32 |
| learning rate | `5e-5` |
| optimizer | AdamW |
| weight decay | 0.01 |
| 최대 epoch | 30 |
| early stopping patience | 7 |

이 값들은 윤주 코드의 SPU-BERT 구조를 기반으로 하지만, 현재 Gazebo 학습 실험을 위해 명시적으로 정리한 설정이다.

---

## 12. 10단계: Gazebo-only 모델 학습

## 12.1 목적

ETH/UCY 사전학습 없이, 현재 수집한 1,493개 Gazebo sample만으로 처음부터 학습한다.

```text
Random initialization
        ↓
Gazebo train 1,044 samples
        ↓
validation 224 samples로 checkpoint 선택
        ↓
test 225 samples로 최종 평가
```

설정:

```text
configs/spubert/moai_social_nav_ext_scene_guided_fs.yaml
```

실행:

```bash
cd /home/kistmnl/social_nav/hunavsim_containers
./scripts/train.sh
```

checkpoint:

```text
.../output/spubert_moai_gazebo_guided_mgp_baseline_map20m/model_best.pth
```

**주의:** `output/`의 큰 checkpoint는 Git에서 제외되어 있으므로 다른 PC로 옮길 때 별도 복사해야 한다.

---

## 13. 11단계: ETH/UCY 사전학습 후 Gazebo fine-tuning

Gazebo-only 학습과 비교하기 위해 두 번째 경로도 수행했다.

```text
ETH / UCY 9개 보행자 scene
        │
        ▼
masked trajectory + scene map 사전학습
        │ encoder weight
        ▼
Gazebo guidance-conditioned MGP/TGP fine-tuning
```

## 13.1 ETH/UCY 통합 데이터 준비

스크립트:

```text
.../scripts/prepare_ethucy_all_pretrain.py
```

- 9개 scene
- 원본 row 수: 1,216,620
- 중복 row: 0
- Dataset loader가 생성한 학습 window: 60,831
- trajectory token shape: `(57, 2)`
- map: `(32, 32)`
- map patch: `(4, 256)`
- map 값: `0 / 1 / 2`

ETH/UCY에는 Gazebo의 Nav2 guidance가 없으므로 이 단계는 `guidance_conditioned: false`이다. 그래서 Gazebo MGP 58 token보다 guidance 1개가 적은 57 token이다.

## 13.2 ETH/UCY pretraining

설정:

```text
configs/spubert/ethucy_all_scene_pretrain.yaml
```

실행:

```bash
cd /home/kistmnl/social_nav/hunavsim_containers

MOAI_CONFIG=configs/spubert/ethucy_all_scene_pretrain.yaml \
./scripts/train.sh
```

결과:

- 20 epochs
- masked trajectory pretraining loss: `0.628304 → 0.074494`
- checkpoint 크기: 약 `44.99 MB`

checkpoint:

```text
.../output/spubert_ethucy_all_scene_pretrain/pretrain_model_best.pth
```

## 13.3 Gazebo fine-tuning

설정:

```text
configs/spubert/moai_social_nav_ext_scene_guided_ethucy_pt.yaml
```

여기서:

```yaml
train_mode: pt
pretrain_checkpoint: ./output/spubert_ethucy_all_scene_pretrain/pretrain_model_best.pth
```

를 사용한다.

실행:

```bash
cd /home/kistmnl/social_nav/hunavsim_containers

MOAI_CONFIG=configs/spubert/moai_social_nav_ext_scene_guided_ethucy_pt.yaml \
./scripts/train.sh
```

`SPU-BERT/spubert/training.py`의 `SBertPlusFTTrainer` (`260`행 부근)가 pretrain checkpoint를 읽고, 호환되는 pretrained encoder weight를 MGP/TGP 쪽에 초기화한다.

중요한 해석:

- ETH/UCY pretraining은 사람의 이동과 상호작용, 장면 map 표현을 미리 학습한다.
- ETH/UCY에는 현재 Gazebo의 8 m guidance-conditioned goal task가 없다.
- 따라서 MGP guidance head는 Gazebo fine-tuning에서 충분히 적응해야 한다.

---

## 14. 12단계: 데이터와 모델 입력 검증

검증 스크립트:

```text
.../scripts/validate_gazebo_guided_dataset.py
```

`validate_sample()` (`52`행 부근)이 한 sample마다 다음을 검사한다.

### 14.1 shape 검증

```text
mgp_spatial_ids  == (58, 2)
tgp_spatial_ids  == (58, 2)
traj_lbl         == (12, 2)
goal_lbl         == (2,)
guidance_lbl     == (2,)
env_spatial_ids  == (4, 256)
envs             == (32, 32)
envs_params      == (6,)
```

### 14.2 값 검증

- tensor 안에 NaN 없음
- tensor 안에 Inf 없음
- MGP의 guidance token 위치 `index 21`이 실제 `guidance_lbl`과 동일
- map 값이 허용된 class로 인코딩됨
- guidance가 local map 범위 안에 있음
- 정답 endpoint가 local map 범위 안에 있음

### 14.3 split 검증

- train/val/test 모두 동일한 input shape
- `(recording_id, episode_id)` 중복 0
- 총 sample 수 일치

현재 결과:

```text
1,493 / 1,493 samples:
  finite tensor
  동일 shape
  guidance token contract 통과
  guidance outside map = 0
  endpoint outside map = 0
  episode leakage = 0
```

별도로 guidance 후보 선택과 map 판정에 대한 단위 테스트 **9개가 모두 통과**했다.

---

## 15. 13단계: 평가 지표를 읽는 방법

단순히 trajectory 오차만 보면 지도 안전성을 놓칠 수 있어 두 종류의 지표를 함께 사용했다.

## 15.1 궤적 정확도

### ADE — Average Displacement Error

12개 예측점과 정답점 거리의 평균이다.

\[
ADE = \frac{1}{12}\sum_{t=1}^{12}\|\hat p_t-p_t\|
\]

낮을수록 전체 궤적 모양이 정답과 가깝다.

### FDE — Final Displacement Error

마지막 12번째 예측점과 정답 endpoint의 거리이다.

\[
FDE = \|\hat p_{12}-p_{12}\|
\]

낮을수록 최종 도착 위치가 정확하다.

### GDE — Guidance Displacement Error

선택된 goal과 guidance point 사이 거리이다.

\[
GDE = \|\hat g-g_{guidance}\|
\]

낮을수록 MGP가 guidance 방향에 가까운 goal을 선택했다.

## 15.2 지도 안전성

| 지표 | 의미 |
|---|---|
| Safe candidate rate | MGP가 만든 전체 후보 중 map-valid 후보 비율 |
| Selected-goal valid rate | 최종 선택 goal이 free cell인 sample 비율 |
| Trajectory map-safe rate | 예측 trajectory가 occupied/unknown/out-of-map을 피한 비율 |
| Execution-valid rate | 선택 goal과 전체 trajectory가 모두 실행 가능하다고 판정된 비율 |
| All candidates invalid | 해당 sample에서 MGP 후보가 전부 무효였던 횟수 |

지도 판정은 `model.py`의 `classify_map_points()` (`34`행 부근), 선택은 `select_guided_goal_candidates()` (`104`행 부근)에서 수행한다.

---

## 16. 현재 실험 결과

두 모델은 같은 Gazebo split, 구조, optimizer, learning rate, batch size, seed, 최대 epoch를 사용했다.

| Test metric | Gazebo-only | ETH/UCY pretrain → Gazebo | 변화 |
|---|---:|---:|---:|
| ADE | 1.0208 m | 0.9513 m | **6.8% 개선** |
| FDE | 2.1463 m | 2.0063 m | **6.5% 개선** |
| GDE | **1.4466 m** | 1.6598 m | 14.7% 악화 |
| Safe candidate rate | **95.8%** | 90.9% | -4.9%p |
| Selected-goal valid rate | **98.2%** | 94.7% | -3.6%p |
| Trajectory map-safe rate | 91.6% | **95.6%** | +4.0%p |
| Execution-valid rate | 89.8% | **90.2%** | +0.4%p |
| All candidates invalid | **4 / 225** | 12 / 225 | pretrained 쪽이 많음 |

## 16.1 결과 해석

ETH/UCY 사전학습은:

- 전체 trajectory 모양(ADE)을 개선했다.
- 최종 trajectory endpoint(FDE)를 개선했다.
- TGP의 map-safe trajectory 비율을 높였다.

하지만:

- guidance와 선택 goal의 거리(GDE)는 악화했다.
- MGP goal 후보의 지도 유효성은 낮아졌다.
- 모든 후보가 무효인 test sample도 4개에서 12개로 늘었다.

즉,

> **ETH/UCY 사전학습은 보행자 상호작용과 trajectory encoder에는 도움을 주었지만, Gazebo 전용 guidance-conditioned MGP goal 생성은 직접 가르치지 못했다.**

따라서 “pretraining이 무조건 전체 모델을 더 좋게 만들었다”가 아니라, **trajectory 예측은 좋아졌고 guidance-conditioned goal prediction은 추가 개선이 필요하다**고 해석해야 한다.

---

## 17. 결과 시각화 파일

모델 프로젝트 기준:

```text
gazebo_classic/hunav_gz_classic_ws/src/moai_social_bert_refactored/figures/
```

주요 폴더:

```text
figures/
├── gazebo_guided_mgp_baseline_map20m/
│   └── Gazebo-only 학습/평가 결과
├── ethucy_all_scene_pretrain/
│   └── ETH/UCY pretraining loss 결과
├── gazebo_guided_mgp_ethucy_pretrained/
│   └── ETH/UCY → Gazebo fine-tuning 결과
└── model_comparison/
    ├── 05_gazebo_baseline_vs_ethucy_pretrained.png
    ├── gazebo_baseline_vs_ethucy_pretrained.json
    └── README.md
```

비교 그림 생성 코드:

```text
scripts/compare_guided_experiments.py
```

학습 log 그림 생성 코드:

```text
scripts/visualize_training_log.py
```

---

## 18. 재현용 실행 순서

아래는 처음부터 다시 실행할 때의 논리적 순서이다.

## 18.1 Gazebo 데이터 수집

```bash
cd /home/kistmnl/social_nav/hunavsim_containers

HUNAV_AUTO_GOAL=True \
HUNAV_AUTO_GOAL_MAX_GOALS=0 \
./scripts/run_simulation.sh corridor_run_001
```

여러 map/density/seed 조합으로 이름을 바꾸어 반복한다.

## 18.2 각 recording 후처리

`prepare_dataset.sh`에 **원본 PKL, map 이름, 출력 run 이름** 순서로 전달한다.

```bash
./scripts/prepare_dataset.sh \
  gazebo_classic/hunav_gz_classic_ws/moai_recordings/corridor_run_001.pkl \
  corridor_low \
  corridor_run_001
```

두 번째 인자인 map 이름은 **해당 recording을 실제로 수집한 map과 반드시 같아야 한다.** 다른 map을 넣으면 로봇/보행자가 벽을 통과하는 것처럼 시각화되고, map 안전성 label도 틀려진다. 세 번째 인자는 결과를 구분할 run 이름이며 생략하면 원본 PKL 파일명을 사용한다.

이 wrapper는:

1. 원본 recording 후처리
2. clean sample 저장
3. 사용 가능한 전체 run 병합
4. episode-safe train/val/test split

을 수행한다.

## 18.3 데이터 검증

`prepare_dataset.sh`의 마지막 `[3/3]` 단계가 Docker 안에서 전체 split을 자동 검증하므로, 일반적으로 별도 명령은 필요하지 않다.

검증만 다시 실행하려면 저장소 루트에서 다음과 같이 동일한 data mount를 사용한다.

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  --volume "$PWD/gazebo_classic/hunav_gz_classic_ws/src/moai_social_bert_refactored/data/processed/gazebo/splits_episode:/data/gazebo:ro" \
  moai-social-bert:cu124 \
  python scripts/validate_gazebo_guided_dataset.py \
  --config configs/spubert/moai_social_nav_ext_scene_guided_fs.yaml
```

## 18.4 Gazebo-only baseline

```bash
cd /home/kistmnl/social_nav/hunavsim_containers
./scripts/train.sh
```

## 18.5 ETH/UCY pretraining

```bash
MOAI_CONFIG=configs/spubert/ethucy_all_scene_pretrain.yaml \
./scripts/train.sh
```

## 18.6 ETH/UCY pretrained → Gazebo fine-tuning

```bash
MOAI_CONFIG=configs/spubert/moai_social_nav_ext_scene_guided_ethucy_pt.yaml \
./scripts/train.sh
```

---

## 19. 처음 보는 사람이 자주 묻는 질문

### Q1. 윤주 모델 입력 포맷은 정확히 어디서 확인하는가?

가장 직접적인 기준은:

```text
SPU-BERT/spubert/datasets/moai_social_nav_extended_goal.py
```

이다.

- `__getitem__()`에서 최종 dictionary key 확인
- `build_mgp_tgp_streams()`에서 MGP/TGP token 순서 확인
- `build_local_map_streams()`에서 map patch 입력 확인

모델이 이 입력을 실제로 어떻게 사용하는지는:

```text
SPU-BERT/spubert/model.py
SPU-BERT/spubert/training.py
```

에서 확인한다.

### Q2. 윤주 코드 전체가 있어야 데이터만 변환할 수 있는가?

입력 specification만 정확히 알면 별도 변환기를 작성할 수는 있다. 하지만 실제 호환성을 보장하려면 Dataset adapter뿐 아니라 model forward/inference와 training batch 코드까지 확인해야 한다.

현재는 윤주 repository의 데이터셋, 모델, 학습 코드를 함께 가져와 실제 `DataLoader → model` 경로로 검증했기 때문에 shape만 흉내 낸 것보다 신뢰도가 높다.

### Q3. final_goal과 guidance_point는 같은가?

항상 같지 않다.

- final goal이 8 m 이내면 같다.
- 8 m보다 멀면 guidance는 현재 위치에서 final goal 방향으로 8 m 앞의 중간점이다.

### Q4. target future의 마지막 점과 guidance도 같은가?

아니다.

- `goal_lbl`: 실제 데이터에서 12 step 뒤 로봇 위치
- `guidance_lbl`: Nav2 final goal 방향 최대 8 m의 장기 guidance

MGP는 과거/사람/map/guidance를 보고 “12 step 뒤 도달할 여러 가능 goal”을 예측한다.

### Q5. map의 unknown을 모델이 받는가?

받는다. 현재 token 값 `0`으로 unknown과 padding을 표현한다. `reject_unknown_goals: true`이므로 추론 시 unknown 위 goal 후보는 안전 후보에서 제외한다.

### Q6. Gazebo 수집 중 이미 MGP 후보를 만들고 TGP로 가는가?

아니다. 수집 중 로봇은 Nav2가 움직인다. MGP 후보 생성과 TGP 추론은 저장된 데이터를 모델에 넣는 학습/오프라인 추론 단계에서 수행된다.

---

## 20. 현재 결과의 한계

현재 결과는 의미 있는 첫 비교이지만 최종 연구 결론으로 보기에는 다음 한계가 있다.

1. Gazebo sample이 1,493개로 대규모 학습에는 아직 적다.
2. test는 225 sample이며 동일 seed의 단일 실험이다.
3. ETH/UCY pretraining의 validation은 간소화되어 있어, 최종 transfer 성능이 더 중요한 판단 기준이다.
4. 현재 map은 20 m 범위를 확보한 대신 0.625 m/cell로 정밀도가 낮을 수 있다.
5. 오프라인 예측 안전성은 확인했지만 실제 Gazebo closed-loop 성공률은 아직 측정하지 않았다.
6. 지도상 free라고 해도 로봇 footprint와 costmap inflation까지 완전히 반영한 동역학적 실행 가능성을 보장하지 않는다.
7. 자동 goal/Nav2 demonstration의 행동 편향을 모델이 그대로 배울 수 있다.

---

## 21. 다음에 진행할 우선순위

### 1순위: guidance-conditioned MGP 개선

현재 pretraining 모델은 trajectory는 좋아졌지만 GDE와 candidate validity가 악화했다.

추천:

- pretrained encoder에는 더 작은 learning rate
- 새로 학습하는 Gazebo MGP goal head에는 더 큰 learning rate
- 초기 몇 epoch는 trajectory encoder 일부 freeze
- guidance loss 또는 map-invalid goal penalty 추가

### 2순위: 데이터 확대

맵 이름만 늘리는 것보다 다음 상황의 균형이 중요하다.

- corridor 마주침
- doorway 양보
- intersection 교차
- 사람 밀도 low / medium / high
- 로봇 좌우 회피
- 정지/추월/정면 접근
- 다양한 시작점과 진행 방향

각 map/density마다 여러 random seed와 여러 독립 episode를 모아야 한다.

### 3순위: 반복 실험

최소 3개 seed로 반복하여 평균±표준편차를 제시한다.

```text
seed 20 / 21 / 22
```

### 4순위: map 해상도 ablation

```text
20 m, 32×32  vs  20 m, 64×64
```

를 비교하여 작은 장애물 표현과 계산량 trade-off를 확인한다.

### 5순위: Gazebo closed-loop 연결

최종적으로는:

```text
실시간 robot/human/map
        ↓
adapter
        ↓
guided MGP → valid goal → TGP
        ↓
Nav2 local goal 또는 controller
        ↓
실제 이동 결과 평가
```

로 연결해야 한다.

평가 항목:

- goal 도달 성공률
- 충돌률
- 최소 사람 거리
- SPL/path efficiency
- 정지 시간
- navigation time
- social discomfort 지표

---

## 22. 최종 정리

현재까지 완료한 핵심 결과는 다음과 같다.

1. **Gazebo/HuNavSim/Nav2 자동 주행과 episode 기반 logger를 연결했다.**
2. **로봇 과거 8 step, 미래 12 step, 주변 보행자, final goal, 8 m guidance point를 저장했다.**
3. **품질 필터로 7개 recording에서 1,493개의 clean social sample을 추출했다.**
4. **겹치는 sliding window가 train/test에 섞이지 않도록 episode-safe split을 구현했다.**
5. **Gazebo 세계 좌표 데이터를 윤주 SPU-BERT의 MGP/TGP/map token dictionary로 변환했다.**
6. **8 m guidance와 맞지 않던 8 m local map 문제를 찾아 20 m map으로 수정했다.**
7. **교수님이 제안한 guidance-conditioned MGP → map filtering → guidance 최근접 goal → TGP 흐름을 코드로 연결했다.**
8. **전체 sample의 shape, finite 값, map 범위, guidance token, split leakage를 검증했다.**
9. **Gazebo-only 학습과 ETH/UCY 사전학습 후 Gazebo fine-tuning을 실제로 수행했다.**
10. **사전학습은 ADE/FDE와 trajectory map safety를 개선했지만, guidance goal 품질은 추가 개선이 필요하다는 구체적인 결과를 얻었다.**

따라서 이 프로젝트는 단순히 “데이터 형식을 바꾼 작업”을 넘어,

> **시뮬레이터 데이터 수집 → 데이터 품질 관리 → 좌표·map token 변환 → guidance-conditioned goal/trajectory 학습 → 지도 기반 안전성 평가**

가 하나의 재현 가능한 연구 파이프라인으로 연결된 상태이다.
