#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
run_name="${1:-dual_route_crossing_runtime_smoke}"

export HUNAV_SCENARIO=agents_training_dual_route_joint_crossing.yaml
export HUNAV_GZPOSE_X=-9.0
export HUNAV_GZPOSE_Y=0.0
export HUNAV_GZPOSE_YAW=0.0
export HUNAV_AUTO_GOAL=True
export HUNAV_AUTO_GOAL_MODE=waypoint
export HUNAV_AUTO_GOAL_WAYPOINTS='-9.0,0.0;9.0,0.0'
export HUNAV_AUTO_GOAL_MAX_GOALS="${HUNAV_EVAL_GOALS:-2}"

export HUNAV_ROBOT_PATH_PLANNER=spubert
export HUNAV_ROBOT_SPUBERT_CONFIG_PATH="${HUNAV_ROBOT_SPUBERT_CONFIG_PATH:-/home/hunav_gz_classic_ws/src/moai_social_bert_refactored/configs/spubert/moai_social_nav_ext_scene_guided_gazebo5952_continue_b21.yaml}"
export HUNAV_ROBOT_SPUBERT_CHECKPOINT="${HUNAV_ROBOT_SPUBERT_CHECKPOINT:-/home/hunav_gz_classic_ws/src/moai_social_bert_refactored/output/spubert_gazebo_5952_continue_b21/model_best.pth}"
export HUNAV_ROBOT_SPUBERT_TGP_TOP_K=5
export HUNAV_ROBOT_SPUBERT_FALLBACK_NAV2=True

# Keep the challenge reproducible: pedestrians follow their routes and the
# robot is responsible for avoiding them.
export HUNAV_PEDESTRIANS_AVOID_ROBOT=False
export HUNAV_USE_EVALUATOR=False
export HUNAV_USE_GAZEBO_GUI="${HUNAV_USE_GAZEBO_GUI:-True}"
export HUNAV_USE_RVIZ="${HUNAV_USE_RVIZ:-True}"

exec "${repo_root}/scripts/run_simulation.sh" "${run_name}"
