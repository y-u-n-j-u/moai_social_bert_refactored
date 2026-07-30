#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sim_root="${repo_root}/gazebo_classic"
run_name="${1:-pmb2_$(date +%Y%m%d_%H%M%S)}"

if [[ ! "${run_name}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Run name may contain only letters, numbers, dot, underscore, and hyphen." >&2
  exit 2
fi
if ! docker image inspect pmb2_hunavsim >/dev/null 2>&1; then
  echo "Image pmb2_hunavsim is missing. Run ./scripts/setup.sh first." >&2
  exit 2
fi

mkdir -p "${sim_root}/hunav_gz_classic_ws/moai_recordings"

export HUNAV_ROBOT_TYPE="${HUNAV_ROBOT_TYPE:-pmb2}"
export HUNAV_ROBOT_NAME="${HUNAV_ROBOT_NAME:-pmb2}"
export HUNAV_NAVIGATION="${HUNAV_NAVIGATION:-True}"
export HUNAV_AGENT_MOTION_MODEL="${HUNAV_AGENT_MOTION_MODEL:-hunav}"
export HUNAV_ROBOT_PATH_PLANNER="${HUNAV_ROBOT_PATH_PLANNER:-nav2}"
export HUNAV_ROBOT_SAVE_TRAINING_PKL="${HUNAV_ROBOT_SAVE_TRAINING_PKL:-True}"
export HUNAV_ROBOT_TRAINING_PKL_PATH="/home/hunav_gz_classic_ws/moai_recordings/${run_name}.pkl"
# HuNav actors are teleported to each computed pose at this rate. 30 Hz keeps
# pedestrian motion visually smooth while the 500 Hz physics loop leaves
# enough CPU headroom for Nav2 and Gazebo.
export HUNAV_UPDATE_RATE="${HUNAV_UPDATE_RATE:-30.0}"
export HUNAV_USE_GAZEBO_GUI="${HUNAV_USE_GAZEBO_GUI:-True}"
export HUNAV_USE_RVIZ="${HUNAV_USE_RVIZ:-True}"
export HUNAV_AUTO_GOAL="${HUNAV_AUTO_GOAL:-False}"
export HUNAV_AUTO_GOAL_MODE="${HUNAV_AUTO_GOAL_MODE:-waypoint}"
export HUNAV_AUTO_GOAL_WAYPOINTS="${HUNAV_AUTO_GOAL_WAYPOINTS:-}"
export HUNAV_AUTO_GOAL_SEED="${HUNAV_AUTO_GOAL_SEED:--1}"
export HUNAV_AUTO_GOAL_MIN_DISTANCE="${HUNAV_AUTO_GOAL_MIN_DISTANCE:-6.0}"
export HUNAV_AUTO_GOAL_MAX_DISTANCE="${HUNAV_AUTO_GOAL_MAX_DISTANCE:-20.0}"
export HUNAV_AUTO_GOAL_CLEARANCE="${HUNAV_AUTO_GOAL_CLEARANCE:-0.55}"
export HUNAV_AUTO_GOAL_MIN_EPISODE_DURATION="${HUNAV_AUTO_GOAL_MIN_EPISODE_DURATION:-12.0}"
export HUNAV_AUTO_GOAL_TIMEOUT="${HUNAV_AUTO_GOAL_TIMEOUT:-60.0}"
export HUNAV_AUTO_GOAL_MAX_GOALS="${HUNAV_AUTO_GOAL_MAX_GOALS:-0}"

echo "Recording name: ${run_name}"
echo "Output: gazebo_classic/hunav_gz_classic_ws/moai_recordings/${run_name}.pkl"
echo "Select a scenario, then send several reachable goals with RViz '2D Goal Pose'."
echo "The tool publishes /goal_pose; do not use the action-based 'Nav2 Goal' tool."
echo "Each new RViz goal becomes a separate dataset episode."
echo "Automatic goals: ${HUNAV_AUTO_GOAL} (mode=${HUNAV_AUTO_GOAL_MODE})"
echo

exec "${sim_root}/run-hunav_gz_classic11_pmb2.bash"
