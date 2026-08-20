#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
run_name="${1:-professor_model_only_demo_$(date +%Y%m%d_%H%M%S)}"

# This scenario requires a mapped detour and schedules four moving pedestrians
# along separate crossing/oncoming lanes. Pedestrians ignore the robot, so
# avoidance is the robot's job and not a side effect of reciprocal avoidance.
export HUNAV_SCENARIO="${HUNAV_SCENARIO:-agents_professor_balanced_demo.yaml}"
export HUNAV_GZPOSE_X=-8.2
export HUNAV_GZPOSE_Y=-1.4
export HUNAV_GZPOSE_YAW=0.0

export HUNAV_AUTO_GOAL=True
export HUNAV_AUTO_GOAL_MODE=waypoint
export HUNAV_AUTO_GOAL_WAYPOINTS='-8.2,-1.4;9.0,-1.4'
export HUNAV_AUTO_GOAL_MAX_GOALS=1
export HUNAV_AUTO_GOAL_TIMEOUT=90
export HUNAV_AUTO_GOAL_NO_PROGRESS_TIMEOUT=30

export HUNAV_ROBOT_PATH_PLANNER=spubert
export HUNAV_ROBOT_SPUBERT_CONFIG_PATH=/home/hunav_gz_classic_ws/src/moai_social_bert_refactored/configs/spubert/moai_social_nav_ext_scene_guided_gazebo5952_continue_b21.yaml
export HUNAV_ROBOT_SPUBERT_CHECKPOINT=/home/hunav_gz_classic_ws/src/moai_social_bert_refactored/output/spubert_gazebo_5952_continue_b21/model_best.pth
export HUNAV_ROBOT_SPUBERT_TGP_TOP_K=5
export HUNAV_ROBOT_SPUBERT_RUNTIME_SEED="${HUNAV_ROBOT_SPUBERT_RUNTIME_SEED:-21}"
export HUNAV_ROBOT_SPUBERT_REJECTION_STREAK_LIMIT=10

# A fallback-free completion is the evidence that SPU-BERT supplied every
# executed path. Keep fallback enabled in deployment, but disabled in this demo.
export HUNAV_ROBOT_SPUBERT_FALLBACK_NAV2=False
export HUNAV_PEDESTRIANS_AVOID_ROBOT=False
export HUNAV_USE_NAVGOAL_TO_START=True
export HUNAV_USE_EVALUATOR=False
# 20 Hz actor updates remain visually smooth. Keep the previously validated
# 500 Hz physics loop; reducing physics frequency changed controller timing.
export HUNAV_UPDATE_RATE=20.0
export HUNAV_PHYSICS_UPDATE_RATE=500.0

# Never mix model-generated demo trajectories into the expert training set.
export HUNAV_ROBOT_SAVE_TRAINING_PKL=False
export HUNAV_USE_GAZEBO_GUI="${HUNAV_USE_GAZEBO_GUI:-True}"
export HUNAV_USE_RVIZ="${HUNAV_USE_RVIZ:-True}"
export HUNAV_RVIZ_CONFIG_PATH=/home/hunav_gz_classic_ws/src/moai_hunav_bridge/rviz/social_bert_human_debug.rviz

echo "Professor demo: 4 goal-synchronized pedestrians, forced static detour, SPU-BERT only"
echo "Runtime seed: ${HUNAV_ROBOT_SPUBERT_RUNTIME_SEED}; fallback: disabled"
echo "After 'Automatic goal publisher completed 1 goals', stop with Ctrl+C and run:"
echo "  ./gazebo_classic/analyze-professor-model-demo.bash ${run_name}"
echo

exec "${repo_root}/scripts/run_simulation.sh" "${run_name}"
