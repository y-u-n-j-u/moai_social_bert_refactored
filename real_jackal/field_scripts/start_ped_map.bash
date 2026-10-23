#!/usr/bin/env bash
# Pedestrians shown ON THE SAVED MAP (the same set-up as a drive, minus everything that can move the robot). Run inside the container.
#   lidar + RealSense colour -> YOLO -> 3D positions -> tracker (perception)
#   + real LIO (FAST-LIVO, image off) + /scan + Nav2 map_server / AMCL / planner on /root/jackal_maps/world_map.yaml
# NOT started: the bridge and the safe_path_tracker (nothing here publishes /j100_0519/cmd_vel, so the robot cannot move).
# After it is up: RViz "2D Pose Estimate" on the map, as for a drive.
source /root/jackal_runtime/scripts/jackal_ros_env.bash || exit 1
source /root/moai_stability_ws/install/setup.bash
source /root/fastlivo_fix_ws/install/setup.bash          # FAST-LIVO with the image buffer cap
RUN=/root/jackal_logs/pedtest_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN"; : > "$RUN/pids"; ln -sfn "$RUN" /root/jackal_logs/drive_latest
start() { local name=$1; shift; setsid nohup "$@" > "$RUN/$name.log" 2>&1 < /dev/null & echo "$name $!" >> "$RUN/pids"; echo "started $name pid $!"; }
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
# real LIO (not the fallback identity odometry) + perception with YOLO, as in a drive but with launch_ped_yolo:=true
start_once perception '[f]ull_stack.launch' ros2 launch mid360_bringup full_stack.launch.py use_sim_time:=false \
  use_fast_livo:=true launch_rviz:=false launch_ped_yolo:=true fast_livo_image_enable:=false \
  fast_livo_lidar_msg_type:=pointcloud2 fast_livo_odom_fallback:=false extrinsic_path:=/root/data/calib/extrinsic.txt
start_once filterpass '[f]ilterpass.py' ros2 run mola_bringup filterpass.py --ros-args -p use_sim_time:=false
start_once scan '[p]ointcloud_to_laserscan_node' ros2 run pointcloud_to_laserscan pointcloud_to_laserscan_node --ros-args \
  -r cloud_in:=/livox/lidar_filtered -r scan:=/scan -p target_frame:=base_link -p transform_tolerance:=0.2 \
  -p min_height:=0.1 -p max_height:=1.5 -p angle_min:=-3.14159 -p angle_max:=3.14159 -p angle_increment:=0.0087 \
  -p scan_time:=0.1 -p range_min:=0.35 -p range_max:=30.0 -p use_inf:=true -p inf_epsilon:=1.0 -p use_sim_time:=false
if ! pgrep -f "[r]sswatch.sh" > /dev/null; then setsid nohup /root/rsswatch.sh > "$RUN/fastlivo_rss.log" 2>&1 < /dev/null & echo "rsswatch $!" >> "$RUN/pids"; fi
sleep 8
start_once nav2 '[n]av2_route_planner.launch' ros2 launch moai_jackal_spubert nav2_route_planner.launch.py map:=/root/jackal_maps/world_map.yaml
[ -f /root/ped_overlay.py ] && start_once overlay '[p]ed_overlay.py' python3 /root/ped_overlay.py
sleep 4   # jobs started last die when a docker exec -it session ends: give them time to detach
echo "run dir: $RUN"
echo "Wait ~60 s (LIO + YOLO warm-up), then in RViz: 2D Pose Estimate on the map."
