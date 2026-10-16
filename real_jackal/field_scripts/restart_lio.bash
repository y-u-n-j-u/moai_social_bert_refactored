#!/usr/bin/env bash
# Restart only the LIO part (FAST-LIVO, /scan conversion, memory watch) - the lidar driver, Nav2 and RViz stay up.
# Do this when FAST-LIVO CPU grows (> ~250 %) or after about an hour. The odom frame restarts at 0, so afterwards:
#   restart the bridge (it keeps old state), then 2D Pose Estimate -> RC 1-2 m -> 2D Goal Pose again.
# Run from the host:  docker exec -i moai_jackal_spubert bash --noprofile --norc -s < restart_lio.bash
source /root/jackal_runtime/scripts/jackal_ros_env.bash >/dev/null
source /root/moai_stability_ws/install/setup.bash
R=$(readlink -f /root/jackal_logs/drive_latest)
echo "$(date +%F_%T) $(date +%s) restart_lio.bash: safety stop before the checks/shutdown" >> /root/jackal_logs/stop_reasons.log
echo "motion: $(timeout 20 ros2 service call /spu_bert/enable_motion std_srvs/srv/SetBool "{data: false}" 2>&1 | grep -o "Motion [a-z]*")"
for n in full_stack filterpass scan; do pid=$(awk -v n="$n" '$1==n{print $2}' "$R/pids" | tail -1); [ -n "$pid" ] && kill -INT -- "-$pid" 2>/dev/null; done
for p in $(ps -eo pid,cmd | grep "[r]sswatch.sh" | awk '{print $1}'); do kill "$p" 2>/dev/null; done
sleep 8
echo "fast-livo still running: $(ps -eo cmd | grep -c '[f]astlivo_mappin')   (must be 0)"
bash /root/start_lio_fixed.bash
sleep 45
for t in /livox/lidar /aft_mapped_to_init /scan; do echo "$t: $(timeout 10 ros2 topic hz $t --window 10 2>&1 | grep -m1 'average rate' || echo NO-DATA)"; done
echo "fast-livo: $(tail -1 "$R/fastlivo_rss.log")"
