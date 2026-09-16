#!/bin/bash
set -e

source /opt/ros/humble/setup.bash
source /root/robot_ws/install/setup.bash
if [[ -n "${MOAI_STABILITY_OVERLAY:-}" ]]; then
  source "$MOAI_STABILITY_OVERLAY"
fi

exec "$@"
