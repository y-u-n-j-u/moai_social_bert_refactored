#!/usr/bin/env bash
source /root/jackal_runtime/scripts/jackal_ros_env.bash || exit 1
source /root/moai_stability_ws/install/setup.bash
RUN=$(readlink -f /root/jackal_logs/drive_latest); G=$(date +%H%M%S)
extra_topics=()
[ -n "${EVIDENCE_JOY_TOPIC:-}" ] && extra_topics+=("$EVIDENCE_JOY_TOPIC")
[ -n "${EVIDENCE_MANUAL_CMD_TOPIC:-}" ] && extra_topics+=("$EVIDENCE_MANUAL_CMD_TOPIC")
setsid nohup ros2 bag record -o "$RUN/bag_$G" /tf /tf_static /amcl_pose /aft_mapped_to_init /scan /ped_tracking /goal_pose \
  /spu_bert/global_path_odom /spu_bert/predicted_path /spu_bert/runtime_status /spu_bert/tracker_status \
  /spu_bert/collision_markers /spu_bert/tracker_diagnostics /livox/imu /j100_0519/cmd_vel /spu_bert/runtime_markers /spu_bert/tracker_lookahead /spu_bert/final_goal_odom /spu_bert/debug_rolling_map /ped_overlay /spu_bert/scan_nohuman /ped_detection \
  /spu_bert/selection_evidence /spu_bert/selection_markers /spu_bert/odom_trail /spu_bert/emergency_stop \
  /spu_bert/cmd_vel_dryrun /j100_0519/platform/cmd_vel_unstamped \
  "${extra_topics[@]}" > "$RUN/bag_$G.log" 2>&1 < /dev/null &
echo "bag $!" >> "$RUN/pids"; echo "bag started bag_$G"
