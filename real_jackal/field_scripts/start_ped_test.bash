#!/usr/bin/env bash
# Stage 0 of the pedestrian work: PERCEPTION ONLY, the robot stands still. Run inside the container.
#   lidar driver + RealSense colour (640x480, 15 Hz, no depth) + YOLO + 3D projection + tracker + overlay
# NOT started (nothing here can drive the robot, and it keeps the load low): FAST-LIVO, Nav2, AMCL, bridge, tracker.
# FAST-LIVO is skipped with fast_livo_imu_enable:=false: the launch then publishes an identity odom/TF (odom->base_link),
# which is correct for a robot that does not move.
source /root/jackal_runtime/scripts/jackal_ros_env.bash || exit 1
source /root/moai_stability_ws/install/setup.bash
RUN=/root/jackal_logs/pedtest_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN"; : > "$RUN/pids"; ln -sfn "$RUN" /root/jackal_logs/drive_latest
start() { local name=$1; shift; setsid nohup "$@" > "$RUN/$name.log" 2>&1 < /dev/null & echo "$name $!" >> "$RUN/pids"; echo "started $name pid $!"; }
# Start a part only when it is not running yet (pressing the pane twice must not create a second copy).
start_once() { local name=$1 pattern=$2; shift 2; if pgrep -f "$pattern" > /dev/null; then echo "$name already running - skipped"; else start "$name" "$@"; fi; }
LIVOX_SHARE="$(ros2 pkg prefix --share livox_ros_driver2)"
start_once livox '[l]ivox_ros_driver2_node' ros2 run livox_ros_driver2 livox_ros_driver2_node --ros-args \
  -p xfer_format:=0 -p multi_topic:=0 -p data_src:=0 -p publish_freq:=10.0 -p output_data_type:=0 \
  -p frame_id:=livox_frame -p "user_config_path:=${LIVOX_SHARE}/config/MID360_config.json" \
  -p cmdline_input_bd_code:=livox0000000001
start_once realsense '[r]ealsense2_camera_node' ros2 launch realsense2_camera rs_launch.py \
  rgb_camera.color_profile:=640,480,15 enable_depth:=false enable_gyro:=false enable_accel:=false \
  enable_infra1:=false enable_infra2:=false pointcloud.enable:=false
sleep 6
start_once perception '[f]ull_stack.launch' ros2 launch mid360_bringup full_stack.launch.py use_sim_time:=false \
  use_fast_livo:=true fast_livo_imu_enable:=false launch_rviz:=false launch_ped_yolo:=true \
  extrinsic_path:=/root/data/calib/extrinsic.txt
[ -f /root/ped_overlay.py ] && start_once overlay '[p]ed_overlay.py' python3 /root/ped_overlay.py
# With a terminal (docker exec -it) the jobs started in the last moments die when the script ends; give them time to detach.
sleep 4
echo "run dir: $RUN"
echo "camera + YOLO need ~30-60 s to warm up (the first YOLO call loads the model on the GPU)."
