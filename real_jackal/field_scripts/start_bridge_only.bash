#!/usr/bin/env bash
# Bridge + safe_path_tracker only (Nav2 is already up from start_ped_map.bash), with the people-only rules config.
# Run in the container AFTER start_ped_map.bash (LIO + camera + YOLO + people tracker + Nav2) and after "2D Pose Estimate".
source /root/jackal_runtime/scripts/jackal_ros_env.bash || exit 1
source /root/moai_stability_ws/install/setup.bash
source /root/fastlivo_fix_ws/install/setup.bash
RUN=$(readlink -f /root/jackal_logs/drive_latest); G=$(date +%H%M%S)
CFG=""; [ -f /root/real_jackal_social005_human05.yaml ] && CFG="config_file:=/root/real_jackal_social005_human05.yaml"
echo "bridge config: ${CFG:-image default}"
pgrep -f "[n]av2_route_planner.launch" >/dev/null || echo "WARNING: Nav2 is not running - run start_ped_map.bash first"
pkill -INT -f "[r]eal_jackal_spubert.launch" 2>/dev/null; sleep 2
setsid nohup ros2 launch moai_jackal_spubert real_jackal_spubert.launch.py $CFG cmd_vel_topic:=/j100_0519/cmd_vel use_cuda:=true require_global_path:=true launch_rviz:=false > "$RUN/spubert_real_$G.log" 2>&1 < /dev/null & echo "spubert_real $!" >> "$RUN/pids"; echo "bridge started (log $RUN/spubert_real_$G.log); motion stays OFF until RUN"
sleep 4
