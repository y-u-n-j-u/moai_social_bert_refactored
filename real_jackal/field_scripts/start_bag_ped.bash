#!/usr/bin/env bash
# Record the pedestrian perception test (run inside the container, after start_ped_test.bash).
# Stop it with SIGINT (the AFTER pane does) and never open a bag while it is recording.
source /root/jackal_runtime/scripts/jackal_ros_env.bash || exit 1
source /root/moai_stability_ws/install/setup.bash
RUN=$(readlink -f /root/jackal_logs/drive_latest); G=$(date +%H%M%S)
setsid nohup ros2 bag record -o "$RUN/bag_$G" /tf /tf_static /aft_mapped_to_init /livox/lidar \
  /camera/camera/color/image_raw/compressed /camera/camera/color/camera_info \
  /ped_yolo/detections2d /ped_detection /ped_tracking /ped_traces > "$RUN/bag_$G.log" 2>&1 < /dev/null &
echo "bag $!" >> "$RUN/pids"; echo "recording to $RUN/bag_$G"
