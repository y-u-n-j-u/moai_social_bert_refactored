#!/usr/bin/env bash
# 자칼 프로세스 전부 종료 (컨테이너는 그대로 둔다).
#   사용: ~/capstone_navi/real_jackal/field_scripts/kill_all.sh            # 종료
#         ~/capstone_navi/real_jackal/field_scripts/kill_all.sh --dry-run  # 종료하지 않고 종료될 프로세스 목록만 보기
# 순서: 움직임 끄기 -> RViz 재시작 래퍼 멈추기 -> 녹화(rosbag) 안전하게 닫기 -> 나머지 전부 INT(10초) -> TERM(4초) -> KILL
# 11-16: robot_marker, 보행자 표시 노드(moai_nav_viz), rviz_retry 래퍼, 상태 창(drive_hud/drive_view)을 목록에 추가하고, 녹화를 먼저 닫도록 순서를 바꿈.
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
docker exec -i "${CONTAINER_NAME:-moai_jackal_spubert}" bash --noprofile --norc -s -- "$DRY" <<'INNER'
DRY="$1"
source /root/jackal_runtime/scripts/jackal_ros_env.bash >/dev/null 2>&1; source /root/moai_stability_ws/install/setup.bash
PAT="livox|nav2|full_stack|filterpass|pointcloud_to_laserscan|static_transform_publisher|fastlivo|real_jackal_spubert|safe_path_tracker|odom_trail|robot_marker|selection_view|drive_hud|drive_view|amcl|planner_server|map_server|lifecycle_manager|controller_server|bt_navigator|behavior_server|waypoint_follower|smoother_server|velocity_smoother|rsswatch|rviz2|rviz_retry|ros2 bag|monitor_run|pedestrian_tracker|pedestrian_figures|pedestrian_traces|moai_nav_viz|mid360|mola|realsense|rs_launch|ped_yolo|ped_overlay|tracks_viz|traces_viz|pcloud_detections|topic_connection"
alive() { ps -eo pid,stat,cmd | grep -E "$PAT" | grep -v -E "grep|ps -eo| Z" | awk '{print $1}'; }
names() { for p in "$@"; do ps -o pid=,cmd= -p "$p" 2>/dev/null | cut -c1-110; done; }
if [ "$DRY" = 1 ]; then
  echo "[dry-run] 종료될 프로세스 $(alive | wc -l)개 (아무것도 종료하지 않음):"; names $(alive)
  exit 0
fi
echo "$(date +%F_%T) $(date +%s) kill_all.sh: stop before shutdown" >> /root/jackal_logs/stop_reasons.log
echo "1) 움직임 끄기: $(timeout 20 ros2 service call /spu_bert/enable_motion std_srvs/srv/SetBool '{data: false}' 2>&1 | grep -o 'Motion [a-z ]*' | head -1)"
for p in $(pgrep -f "[r]viz_retry.bash"); do kill "$p" 2>/dev/null; done                    # RViz를 종료해도 다시 켜지 않게 래퍼부터
for p in $(pgrep -f "[m]onitor_run.ba"); do kill "$p" 2>/dev/null; done
BAG=$(pgrep -f "[r]os2 bag reco")
if [ -n "$BAG" ]; then
  echo "2) 녹화를 닫는 중..."; kill -INT $BAG 2>/dev/null
  for _ in $(seq 1 20); do pgrep -f "[r]os2 bag reco" >/dev/null || break; sleep 0.5; done
  echo "   녹화 프로세스: $(pgrep -f '[r]os2 bag reco' | wc -l)개 남음"
else echo "2) 녹화 중인 것 없음"; fi
echo "3) 나머지 종료: 대상 $(alive | wc -l)개"
alive | xargs -r kill -INT;  sleep 10
alive | xargs -r kill -TERM; sleep 4
alive | xargs -r kill -KILL; sleep 1
LEFT=$(alive); echo "남은 프로세스: $(echo "$LEFT" | grep -c .)"; [ -n "$LEFT" ] && names $LEFT
INNER
