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
export HUNAV_UPDATE_RATE="${HUNAV_UPDATE_RATE:-10.0}"
export HUNAV_USE_GAZEBO_GUI="${HUNAV_USE_GAZEBO_GUI:-True}"
export HUNAV_USE_RVIZ="${HUNAV_USE_RVIZ:-True}"

echo "Recording name: ${run_name}"
echo "Output: gazebo_classic/hunav_gz_classic_ws/moai_recordings/${run_name}.pkl"
echo "Select a scenario, then send several reachable goals with RViz 'Nav2 Goal'."
echo "Each new RViz goal becomes a separate dataset episode."
echo

exec "${sim_root}/run-hunav_gz_classic11_pmb2.bash"
