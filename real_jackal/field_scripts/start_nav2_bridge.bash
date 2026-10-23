#!/usr/bin/env bash
source /root/jackal_runtime/scripts/jackal_ros_env.bash || exit 1
source /root/moai_stability_ws/install/setup.bash
source /root/fastlivo_fix_ws/install/setup.bash
RUN=$(readlink -f /root/jackal_logs/drive_latest); G=$(date +%H%M%S)
# 10-23: the people-only rules (people are not walls, radii 0.50/0.30 m) live in this config copy, not in the image's default file
CFG=""; [ -f /root/real_jackal_social005_human05.yaml ] && CFG="config_file:=/root/real_jackal_social005_human05.yaml"
echo "bridge config: ${CFG:-image default}"
setsid nohup ros2 launch moai_jackal_spubert nav2_route_planner.launch.py map:=/root/jackal_maps/world_map.yaml > "$RUN/nav2_$G.log" 2>&1 < /dev/null & echo "nav2 $!" >> "$RUN/pids"; echo "nav2 $!"
sleep 8
setsid nohup ros2 launch moai_jackal_spubert real_jackal_spubert.launch.py $CFG cmd_vel_topic:=/j100_0519/cmd_vel use_cuda:=true require_global_path:=true launch_rviz:=false > "$RUN/spubert_real_$G.log" 2>&1 < /dev/null & echo "spubert_real $!" >> "$RUN/pids"; echo "spubert_real $!"
