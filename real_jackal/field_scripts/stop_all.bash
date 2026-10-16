#!/usr/bin/env bash
# Stop everything that was started for a drive (inside the container), in a safe order. The container itself stays up.
# Run from the host:  docker exec -i moai_jackal_spubert bash --noprofile --norc -s < stop_all.bash
source /root/jackal_runtime/scripts/jackal_ros_env.bash >/dev/null
source /root/moai_stability_ws/install/setup.bash
R=$(readlink -f /root/jackal_logs/drive_latest)
echo "$(date +%F_%T) $(date +%s) stop_all.bash: safety stop before the checks/shutdown" >> /root/jackal_logs/stop_reasons.log
echo "motion: $(timeout 20 ros2 service call /spu_bert/enable_motion std_srvs/srv/SetBool "{data: false}" 2>&1 | grep -o "Motion [a-z]*")"
stop() { for n in "$@"; do pid=$(awk -v n="$n" '$1==n{print $2}' "$R/pids" | tail -1); [ -n "$pid" ] && kill -INT -- "-$pid" 2>/dev/null && echo "stop signal: $n (pid $pid)"; done; }
for p in $(ps -eo pid,cmd | grep "[m]onitor_run.ba" | awk '{print $1}'); do kill "$p" 2>/dev/null; done
pkill -INT -f "ros2 bag reco[r]d" 2>/dev/null
stop spubert_real rviz odom_trail; sleep 5     # drive side
stop nav2; sleep 4                             # Nav2
stop full_stack filterpass scan; sleep 6       # LIO
for p in $(ps -eo pid,cmd | grep "[r]sswatch.sh" | awk '{print $1}'); do kill "$p" 2>/dev/null; done
stop livox; sleep 4                            # lidar driver
echo "cmd_vel publishers: $(timeout 12 ros2 topic info /j100_0519/cmd_vel 2>&1 | grep 'Publisher count')"
left=$(ps -eo pid,stat,cmd | grep -E "[r]eal_jackal_spubert_bridge|[s]afe_path_tracker|[f]astlivo_mappin|[l]ivox_ros|[a]mcl|[p]lanner_server|[o]dom_trail" | grep -vc " Z")
echo "processes of ours still running: $left"
