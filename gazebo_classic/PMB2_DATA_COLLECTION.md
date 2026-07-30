# PMB2 Gazebo data collection

This branch can collect the robot target, surrounding HuNavSim pedestrians,
the active RViz goal, and 20-step trajectory windows while PMB2 runs with Nav2.
The learned robot checkpoint is not required for this collection mode.

## End-to-end commands

Repository root의 통합 명령을 사용하는 것이 가장 간단하다.

```bash
./scripts/setup.sh
./scripts/run_simulation.sh pmb2_run_001
./scripts/prepare_dataset.sh \
  gazebo_classic/hunav_gz_classic_ws/moai_recordings/pmb2_run_001.pkl \
  compact_corridor
./scripts/validate.sh
./scripts/train.sh
```

아래 내용은 각 단계를 직접 실행할 때의 세부 명령이다.

## 1. Build the image

```bash
cd /home/kistmnl/social_nav/hunavsim_containers/gazebo_classic
docker build \
  -t pmb2_hunavsim \
  -f Dockerfile.hunav_gz_classic11_pmb2 \
  .
```

## 2. Run Gazebo, Nav2, and the dataset logger

```bash
cd /home/kistmnl/social_nav/hunavsim_containers/gazebo_classic

HUNAV_ROBOT_TYPE=pmb2 \
HUNAV_ROBOT_NAME=pmb2 \
HUNAV_NAVIGATION=True \
HUNAV_AGENT_MOTION_MODEL=hunav \
HUNAV_ROBOT_PATH_PLANNER=nav2 \
HUNAV_ROBOT_SAVE_TRAINING_PKL=True \
HUNAV_ROBOT_TRAINING_PKL_PATH=/home/hunav_gz_classic_ws/moai_recordings/pmb2_run_001.pkl \
HUNAV_UPDATE_RATE=10.0 \
./run-hunav_gz_classic11_pmb2.bash
```

Choose a scenario from the terminal menu. After Gazebo and RViz open, use
RViz `Nav2 Goal` to send a goal. Every new RViz goal starts a new episode and
clears the unfinished 20-frame window, so samples do not cross goal changes.

The container bind-mounts the workspace, so the raw file is written on the
host at:

```text
gazebo_classic/hunav_gz_classic_ws/moai_recordings/pmb2_run_001.pkl
```

## 3. Add the local map and create model samples

While the simulation container is running, open a second terminal:

```bash
cd /home/kistmnl/social_nav/hunavsim_containers/gazebo_classic
./ros2_hunavsim.bash
```

Then run this command inside the container, changing the map YAML to match the
scenario selected in step 2:

```bash
python3 /home/hunav_gz_classic_ws/src/moai_hunav_bridge/scripts/postprocess_pedestrian_dataset.py \
  --input /home/hunav_gz_classic_ws/moai_recordings/pmb2_run_001.pkl \
  --map-yaml /home/hunav_gz_classic_ws/src/hunav_gazebo_wrapper/maps/compact_corridor.yaml \
  --out-dir /home/hunav_gz_classic_ws/moai_recordings/processed/pmb2_run_001 \
  --name pmb2_run_001 \
  --map-size-m 8.0 \
  --map-grid-size 32 \
  --guidance-radius 8.0
```

The processed samples contain:

- `target_past` `(8, 2)` and `target_future` `(12, 2)`
- `neighbor_past` `(N, 8, 2)` and `neighbor_future` `(N, 12, 2)`
- `final_goal` and the maximum-8-m `guidance_point`
- robot-centered, heading-aligned `local_map` `(32, 32)`
- `episode_id` for split isolation

Use a unique output name for every simulation run. Split by episode/run rather
than randomly splitting overlapping windows.

At least three independent RViz-goal episodes are required to create non-empty
train, validation, and test partitions. The integrated dataset command assigns
the complete `(recording_id, episode_id)` group to exactly one partition.
