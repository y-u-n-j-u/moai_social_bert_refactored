#!/usr/bin/env bash
# Start RViz with a config and keep it running. Two failures happen at random with software rendering in the container:
#  - right after start "Could not load display config: Invalid argument" (an empty RViz opens)
#  - RViz exits by itself (crash) a few seconds after loading
# Both are retried (max 8 starts). Usage: rviz_retry.bash <config.rviz>   (log: <run dir>/rviz_retry.log)
source /root/jackal_runtime/scripts/jackal_ros_env.bash >/dev/null 2>&1
source /root/moai_stability_ws/install/setup.bash
export LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe
CFG="${1:-/root/real_jackal_ped.rviz}"
R=$(readlink -f /root/jackal_logs/drive_latest)
for try in 1 2 3 4 5 6 7 8; do
  START=$(date +%s)
  rviz2 -d "$CFG" > "$R/rviz.log" 2>&1 &
  PID=$!
  sleep 7
  if grep -q "Could not load display config" "$R/rviz.log"; then
    echo "$(date +%T) try $try: config did not load - restarting" >> "$R/rviz_retry.log"
    kill $PID 2>/dev/null; sleep 1; kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
    continue
  fi
  echo "$(date +%T) try $try: config loaded" >> "$R/rviz_retry.log"
  wait $PID; CODE=$?
  UP=$(( $(date +%s) - START ))
  echo "$(date +%T) try $try: RViz ended with exit code $CODE after $UP s" >> "$R/rviz_retry.log"
  [ $UP -ge 60 ] && exit 0            # it ran for a while: closed by the user, do not restart
done
echo "$(date +%T) gave up" >> "$R/rviz_retry.log"
