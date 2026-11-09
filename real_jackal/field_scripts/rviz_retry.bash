#!/usr/bin/env bash
# Start RViz with a config and keep it running. Two failures happen at random with software rendering in the container:
#  - right after start "Could not load display config: Invalid argument" (an empty RViz opens)
#  - RViz exits by itself (crash) a few seconds after loading, or hangs (deadlock, ~0 CPU)
# Both are retried (max 8 starts); a third case (RViz not subscribed to the camera) is retried too. Usage: rviz_retry.bash <config.rviz>   (log: <run dir>/rviz_retry.log)
source /root/jackal_runtime/scripts/jackal_ros_env.bash >/dev/null 2>&1
source /root/moai_stability_ws/install/setup.bash
export LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe
CFG="${1:-/root/real_jackal_ped.rviz}"
# Only ONE RViz at a time: close every RViz that is already running (a second one fights for the display and the topics).
for P in $(pgrep -f "[r]viz2 -d" ); do kill "$P" 2>/dev/null; done; sleep 2
for P in $(pgrep -f "[r]viz2 -d" ); do kill -9 "$P" 2>/dev/null; done
R=$(readlink -f /root/jackal_logs/drive_latest)
VIEW="$(dirname -- "${BASH_SOURCE[0]}")/selection_view.py"
if [ -f "$VIEW" ] && python3 -c 'import tkinter, rclpy' >> "$R/selection_view.log" 2>&1; then
  setsid nohup python3 -u "$VIEW" >> "$R/selection_view.log" 2>&1 < /dev/null &
  echo "Candidate decisions window requested (log: $R/selection_view.log)."
else
  echo "Candidate decisions window could not start: check $VIEW and $R/selection_view.log (Python Tk/ROS required)." >&2
fi
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
  sleep 5
  TICKS=$(awk '{print $14+$15}' /proc/$PID/stat 2>/dev/null)   # CPU time of RViz so far (100 ticks = 1 s)
  if [ -z "$TICKS" ] || [ "$TICKS" -lt 400 ]; then               # a hung RViz (deadlock at start) uses almost no CPU
    echo "$(date +%T) try $try: RViz hung or died (cpu ticks=${TICKS:-none}) - restarting" >> "$R/rviz_retry.log"
    kill $PID 2>/dev/null; sleep 1; kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
    continue
  fi
  # 11-08: an RViz that looked healthy (CPU ok, config loaded) once never subscribed to the camera / mask topics: both Image panels
  # stayed on "No Image". If the camera is publishing but RViz is not among its subscribers within ~25 s, start over.
  CAM=/camera/camera/color/image_raw; SUBSCRIBED=0; HAVE_CAM=0
  for k in 1 2 3 4 5 6; do
    INFO=$(timeout 8 ros2 topic info -v $CAM 2>/dev/null)
    grep -q "Publisher count: [1-9]" <<< "$INFO" || break                      # no camera running (e.g. dry-run without it): nothing to check
    HAVE_CAM=1
    if awk '/Subscription count/{f=1} f&&/Node name: rviz/{found=1} END{exit !found}' <<< "$INFO"; then SUBSCRIBED=1; break; fi
    sleep 3
  done
  if [ "$HAVE_CAM" = 1 ] && [ "$SUBSCRIBED" = 0 ]; then
    echo "$(date +%T) try $try: RViz is not subscribed to $CAM (Image panels would stay empty) - restarting" >> "$R/rviz_retry.log"
    kill $PID 2>/dev/null; sleep 1; kill -9 $PID 2>/dev/null; wait $PID 2>/dev/null
    continue
  fi
  echo "$(date +%T) try $try: config loaded, RViz working (cpu ticks=$TICKS)" >> "$R/rviz_retry.log"
  wait $PID; CODE=$?
  UP=$(( $(date +%s) - START ))
  echo "$(date +%T) try $try: RViz ended with exit code $CODE after $UP s" >> "$R/rviz_retry.log"
  [ $UP -ge 60 ] && exit 0            # it ran for a while: closed by the user, do not restart
done
echo "$(date +%T) gave up" >> "$R/rviz_retry.log"
