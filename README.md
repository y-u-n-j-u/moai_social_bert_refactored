# Gazebo → guided SPU-BERT end-to-end pipeline

이 브랜치는 하나의 저장소에서 다음 전체 흐름을 실행한다.

```text
HuNavSim + Gazebo + PMB2 + Nav2
  → robot/pedestrian/RViz goal trajectory PKL 수집
  → 8 m guidance point + heading-aligned 32×32 local map 생성
  → RViz-goal episode 단위 train/val/test 분할
  → 윤주 guided SPU-BERT 입력 shape 전수 검증
  → MGP 후보 goal 생성 + map 필터 + TGP trajectory 학습
```

학습 checkpoint는 저장소에 포함하지 않는다. Gazebo 데이터로 from-scratch
학습하면 `model_best.pth`가 로컬 `output/` 아래에 생성된다.

## 권장 환경

- Ubuntu 22.04
- NVIDIA GPU 및 드라이버
- Docker Engine
- NVIDIA Container Toolkit

## 1. Clone

```bash
git clone \
  --branch agent/hunavsim-gazebo-nav2-spubert-integration \
  https://github.com/wnsdnnn/moai_social_bert_refactored.git

cd moai_social_bert_refactored
```

## 2. Docker 환경 구축

```bash
./scripts/setup.sh
```

이 명령은 다음 이미지 두 개를 만든다.

- `pmb2_hunavsim`: ROS 2 Humble + HuNavSim + Gazebo Classic + PMB2 + Nav2
- `moai-social-bert:cu124`: guided SPU-BERT 데이터 검증 및 학습

## 3. Gazebo 실행 및 데이터 수집

```bash
./scripts/run_simulation.sh pmb2_run_001
```

터미널에서 scenario를 선택한 다음 RViz의 `2D Goal Pose`로 도달 가능한 목적지를
여러 번 지정한다. 새로운 RViz goal을 보낼 때마다 새 episode가 시작되므로,
한 번의 실행에서도 최소 3개 이상의 goal episode를 수집해야
train/val/test를 분리할 수 있다.

이 브랜치의 Docker image는 PAL 기본 `Nav2 Goal` 도구를 제거하고
`/goal_pose`를 발행하는 `2D Goal Pose`만 사용한다. `/goal_pose`는 logger와
SPU-BERT/Nav2 bridge가 동시에 받는다. `nav2` 수집 모드에서는 Nav2
`bt_navigator`가 `/goal_pose`를 직접 처리하며, bridge는 중복 action을 보내지
않는다.

RViz에서 goal을 반복해서 지정하지 않고 자동 수집하려면:

```bash
HUNAV_AUTO_GOAL=True \
HUNAV_AUTO_GOAL_MAX_GOALS=20 \
./scripts/run_simulation.sh pmb2_auto_001
```

기본 자동 모드는 선택한 학습 맵의 안전한 중앙 waypoint 두 개를 왕복한다.
`corridor=(-8,0)<->(8,0)`, `doorway=(-5.5,0)<->(5.5,0)`,
`intersection=(-7.5,0)<->(7.5,0)`이다. 모든 goal은 Nav2
`ComputePathToPose`로 실제 경로가 존재하는지 검증한 뒤 `/goal_pose`로
발행한다. 이후 일반화 데이터를 수집할 때만
`HUNAV_AUTO_GOAL_MODE=random`을 지정한다.
자동 모드 중에는 RViz에서 수동 goal을 추가로 보내지 않는다.

학습 맵은 HuNav 보행자를 30 Hz로 갱신하고 Gazebo 물리를 500 Hz
(`max_step_size=0.002`)로 실행한다. PMB2의 Nav2 localization은 대칭 복도와
이동 보행자 때문에 흔들리는 AMCL TF 대신 Gazebo `/ground_truth_odom`으로
`map->odom`을 보정한다. Logger는 목표에 실제 도달한 episode만 확정 저장하며,
timeout·goal 교체·중간 종료 episode는 폐기한다.

화면 없이 장시간 자동 수집하려면 다음처럼 Gazebo GUI와 RViz를 모두 끌 수 있다.

```bash
HUNAV_AUTO_GOAL=True \
HUNAV_AUTO_GOAL_MAX_GOALS=50 \
HUNAV_USE_GAZEBO_GUI=False \
HUNAV_USE_RVIZ=False \
./scripts/run_simulation.sh pmb2_auto_headless_001
```

원본 PKL:

```text
gazebo_classic/hunav_gz_classic_ws/moai_recordings/pmb2_run_001.pkl
```

## 4. 후처리·분할·입력 검증

두 번째 인자는 선택한 scenario가 사용하는 map 이름이다.

```bash
./scripts/prepare_dataset.sh \
  gazebo_classic/hunav_gz_classic_ws/moai_recordings/pmb2_run_001.pkl \
  training_corridor
```

다른 주행을 추가하려면 이름을 바꿔 3번과 4번을 반복한다. 기존에 처리된
모든 주행은 매번 다시 합쳐지고, `(recording_id, episode_id)` 단위로
train/val/test가 생성된다. 따라서 같은 sliding-window episode가 서로 다른
split에 들어가지 않는다.

최종 split:

```text
gazebo_classic/hunav_gz_classic_ws/src/moai_social_bert_refactored/
  data/processed/gazebo/splits_episode/
    pmb2_true_goal_gp_v1_clean_social_train.pkl
    pmb2_true_goal_gp_v1_clean_social_val.pkl
    pmb2_true_goal_gp_v1_clean_social_test.pkl
```

## 5. 전체 검증과 학습

```bash
# 모델 설정과 경로만 확인
./scripts/train.sh --dry_run

# 단위 테스트 + 전 sample 입력 검증 + GPU forward/backward
./scripts/validate.sh

# 기본 200 epoch 학습
./scripts/train.sh
```

학습 결과:

```text
gazebo_classic/hunav_gz_classic_ws/src/moai_social_bert_refactored/
  output/spubert_moai_gazebo_guided_mgp_fs/model_best.pth
```

실행에 문제가 생기면 세부 수집 설명은
[`gazebo_classic/PMB2_DATA_COLLECTION.md`](gazebo_classic/PMB2_DATA_COLLECTION.md),
guided MGP/TGP 입력과 모델 설명은
[`GAZEBO_GUIDED_MGP.md`](gazebo_classic/hunav_gz_classic_ws/src/moai_social_bert_refactored/GAZEBO_GUIDED_MGP.md)를 확인한다.

---

# Original HuNavSim containers

**This is a work in progress version**

This is a package with the files and tools required to build and run Docker containers with the HuNavSim and different Robotics simulators under ROS 2. Te available options are:

1. HuNavSim + Gazebo Classic 11 + ROS 2 Humble + PAL PMB2 robot
2. HuNavSim + Gazebo Fortress   + ROS 2 Humble  (**NO ROBOT FOR THE MOMENT!**)
3. HuNavSim + Isaac Sim         + ROS 2 Humble + different robots
4. HuNavSim + Webots            + ROS 2 Humble + PAL TIAGo robot"

Option 1 Gazebo Classic includes the [PMB2 Robot (ROS 2)](https://github.com/pal-robotics/pmb2_simulation/tree/humble-devel) robot of PAL robotics and the ROS 2 navigation system up.
Option 2 Gazebo Fortress does not include any robot for the moment. We will work to include it. 
Option 3 Isaac Sim allows to select between different robots.
Option 4 Webots includes the TIAGo robot of PAL Robotics ([TIAGO Lite](https://github.com/cyberbotics/webots_ros2/wiki/Example-TIAGo)). 

The containers contains all the required packages to run different simulations of the HuNavSim in the chosen Robotics Simulator. Moreover, the HuNavSim software is installed in a shared directory with the host system, so the user can modify or create new simulations and store them.   


# Dependencies

| Requirement | Notes / Links |
|-------------|---------------|
| **Docker**  | Install Docker Engine following the official guide → <https://docs.docker.com/desktop/setup/install/linux/> |
| **Git**     | Needed to clone the HuNavSim and wrappers repositories |
| **Nvidia container toolkit** | https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html |
| **Isaac Sim container prerequisites** <br>(**only for option 3 – Isaac Sim**) | Before running the *Isaac Sim* container you **must** complete the “Container Setup” steps in the official Isaac Sim docs (NVIDIA driver and Container Toolkit).<br>Guide → <https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html#container-setup> |



# Installation

Once you have installed docker, you can install the system by executing the bash script *install.sh*.
First give execution permission to the script from a terminal:

```sh
chmod +x install.sh
```

Then you can execute it:

```sh
./install.sh
```

This script will ask you about the option that you want to install, and will show the installation progress. 
**The HuNavSim software and the indicated wrapper, will be installed in a shared workspace shared with the docker container, located inside the indicated simulator directory, so you can modify and create your own environments with persistance.**   


# Execution

After system installation, the installation program will show on the screen the name of the script that is required to run the system. Before execute it, we must check the execution permissions (only required the first time).

```sh
chmod +x given_script_name.bash
```

Execute the indicated script:

```sh
./given_script_name.bash
```

## Execution Option 1, 2 and 4 (Gazebo Classic, Gazebo Fortress and Webots)

The previous script will run the docker container and will compile the current workspace. Moreover it will show a menu with the possible options. It will show the stored scenarios that can be executed (.yaml files in the *scenarios* directory of the running wrapper) along with other options. For example:

```sh
========= HuNavSim Docker Menu =========
  1) Run scenario agents_training_corridor_high.yaml
  2) Run scenario agents_training_corridor_low.yaml
  3) Run scenario agents_training_corridor_medium.yaml
  ... doorway/intersection low, medium, high ...
  10) Create a new scenario with RViz
========================================
Select an option (number): 
```

With option 4, the RViz with the HuNavSim panel will be opened so the user can create new scenarios (**ONGOING WORK**).

The user can create/modify the simulations throught the shared workspace. 

NOTE FOR GAZEBO CLASSIC: SOMETIMES, GAZEBO TAKES A LONG TIME TO LAUNCH THE FIRST TIME LEADING TO ERRORS IN THE SYSTEM. IN THAT CASE, STOP THE SYSTEM (CRTL+C), THE MENU WILL SHOW UP AGAIN, AND RE-RUN THE ENVIRONMENT AGAIN. IT SHOULD WORK THE SECOND TIME.

## Execution Option 3 (Isaac Sim)

### Running the Container

To start the container and automatically configure the simulation environment, run:

```bash
./run-hunav_isaac.bash
```

This script will:

- Prompt you for **Omniverse credentials** (only on the first run).
- Automatically build and source the **HuNav Isaac workspace**.
- Launch the Isaac Sim container.

### Omniverse Login (Required)

To connect to the **Nucleus server**, which allows **Isaac Sim** to download and cache asset packs locally, you must log in with your **Omniverse username and password**.

If you don’t have an account, you can register here:\
   [https://developer.nvidia.com/login](https://developer.nvidia.com/login)

Your credentials are securely stored using your system’s **keyring** (`libsecret` via `secret-tool`) after first use.\
You won’t be prompted again in future runs unless you manually reset them.

#### Resetting Credentials (Optional)

If you entered incorrect credentials or want to re-authenticate, you can run:

```bash
./run-hunav_isaac.bash --reset-credentials
```

This will:

- Clear the stored credentials from the system keyring.
- Prompt you again for your username and password.
- Immediately launch the container afterward.

### Inside the Container

Launch the simulation using one of the following options:

```bash
# Interactive launcher (recommended)
hunav_isaac

# Launch with custom configuration
hunav_isaac --config warehouse_agents.yaml --robot carter_ROS

# or use ROS2 command with same arguments (hunav_isaac is an alias for this)
ros2 run hunav_isaac_wrapper hunav_isaac_launcher
```

**Available robots**: `jetbot`, `create3`, `carter`, `carter_ROS`\
**Available worlds**: `warehouse`, `hospital`, `office`

### ⓘ Additional Notes

- The first launch may take **several minutes** while Isaac Sim populates its caches. Subsequent launches are significantly faster thanks to persistent volumes.
- The `Carter_ROS` robot is compatible with the **ROS 2 Nav2** stack for navigation. To control **Carter** robot via **ROS 2 Nav2**:

```bash
# Open a new terminal 
terminator &

# In the new terminal, execute:
cd /workspace/hunav_isaac_ws/src/Hunav_isaac_wrapper/src
ros2 launch carter_navigation carter_navigation.launch.py \
  params_file:="config/navigation_params/carter_navigation_params.yaml" \
  map:="scenarios/occupancy_maps/warehouse.yaml"
```
