#!/bin/bash
set -e

source /opt/ros/humble/setup.bash
source /root/robot_ws/install/setup.bash

exec "$@"
