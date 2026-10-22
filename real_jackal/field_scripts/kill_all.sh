#!/usr/bin/env bash
# 자칼 프로세스 전부 종료 (컨테이너는 유지). 사용: ~/capstone_navi/real_jackal/field_scripts/kill_all.sh
docker exec "${CONTAINER_NAME:-moai_jackal_spubert}" bash --noprofile --norc -c '
source /root/jackal_runtime/scripts/jackal_ros_env.bash >/dev/null; source /root/moai_stability_ws/install/setup.bash
echo "$(date +%F_%T) $(date +%s) kill_all.sh: stop before shutdown" >> /root/jackal_logs/stop_reasons.log
timeout 20 ros2 service call /spu_bert/enable_motion std_srvs/srv/SetBool "{data: false}" >/dev/null 2>&1   # 먼저 정지
PAT="livox|nav2|full_stack|filterpass|pointcloud_to_laserscan|static_transform_publisher|fastlivo|real_jackal_spubert|safe_path_tracker|odom_trail|amcl|planner_server|map_server|lifecycle_manager|controller_server|bt_navigator|behavior_server|waypoint_follower|smoother_server|velocity_smoother|rsswatch|rviz2|ros2 bag|monitor_run|pedestrian_tracker|mid360|mola|realsense|rs_launch|ped_yolo|ped_overlay|tracks_viz|traces_viz|pcloud_detections|topic_connection"
alive() { ps -eo pid,stat,cmd | grep -E "$PAT" | grep -v -E "grep|ps -eo| Z" | awk "{print \$1}"; }
alive | xargs -r kill -INT;  sleep 10
alive | xargs -r kill -TERM; sleep 4
alive | xargs -r kill -KILL; sleep 1
echo "남은 프로세스: $(alive | wc -l)"'
