#!/usr/bin/env bash
source /root/jackal_runtime/scripts/jackal_ros_env.bash || exit 1
source /root/moai_stability_ws/install/setup.bash
RUN=$(readlink -f /root/jackal_logs/fallback_test_latest); G=$(date +%H%M%S)
setsid nohup ros2 bag record -o "$RUN/bag_$G" /tf /tf_static /amcl_pose /aft_mapped_to_init /scan /ped_tracking /goal_pose \
  /spu_bert/global_path_odom /spu_bert/predicted_path /spu_bert/runtime_status /spu_bert/tracker_status \
  /spu_bert/collision_markers /spu_bert/tracker_diagnostics /livox/imu /j100_0519/cmd_vel /joint_states > "$RUN/bag_$G.log" 2>&1 < /dev/null &
echo "bag $!" >> "$RUN/pids"; echo "bag started bag_$G"
