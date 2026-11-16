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
# 11-16: an older bridge + tracker did not always stop after the INT within 2 s, so two trackers ran (cmd_vel had 2 publishers).
# Ask them to stop, wait up to 10 s, then terminate / kill whatever is left; start the new one only when none is running.
pkill -INT -f "[r]eal_jackal_spubert.launch" 2>/dev/null
for _ in $(seq 1 20); do pgrep -f "[r]eal_jackal_spubert_bridge|[s]afe_path_tracker|[r]eal_jackal_spubert.launch" > /dev/null || break; sleep 0.5; done
pkill -TERM -f "[r]eal_jackal_spubert_bridge|[s]afe_path_tracker|[r]eal_jackal_spubert.launch" 2>/dev/null; sleep 2
pkill -KILL -f "[r]eal_jackal_spubert_bridge|[s]afe_path_tracker|[r]eal_jackal_spubert.launch" 2>/dev/null; sleep 1
if pgrep -f "[r]eal_jackal_spubert_bridge|[s]afe_path_tracker" > /dev/null; then echo "ERROR: an old bridge/tracker is still running - not starting a second one" >&2; exit 1; fi
setsid nohup ros2 launch moai_jackal_spubert real_jackal_spubert.launch.py $CFG cmd_vel_topic:=/j100_0519/cmd_vel use_cuda:=true require_global_path:=true launch_rviz:=false > "$RUN/spubert_real_$G.log" 2>&1 < /dev/null & echo "spubert_real $!" >> "$RUN/pids"; echo "bridge started (log $RUN/spubert_real_$G.log); motion stays OFF until RUN"
sleep 4
