#!/usr/bin/env bash
source /root/jackal_runtime/scripts/jackal_ros_env.bash >/dev/null 2>&1; source /root/moai_stability_ws/install/setup.bash
R=$(readlink -f /root/jackal_logs/drive_latest)
timeout 20 ros2 service call /spu_bert/enable_motion std_srvs/srv/SetBool "{data: false}" >/dev/null 2>&1
python3 /root/prep.py 2>&1 | grep -v XMLPARSER | grep '^{' > /root/jackal_logs/last_goal.json; echo "saved goal: $(cat /root/jackal_logs/last_goal.json | cut -c1-120)"
for p in $(ps -eo pid,args | awk '/[r]eal_jackal_spubert.launch|[s]pubert_bridge|[s]afe_path_tracker/ && !/restart_bridge/{print $1}'); do kill -INT $p; done; sleep 5
setsid nohup ros2 launch moai_jackal_spubert real_jackal_spubert.launch.py config_file:=/root/real_jackal_social005_human05.yaml cmd_vel_topic:=/j100_0519/cmd_vel use_cuda:=true require_global_path:=true launch_rviz:=false > $R/spubert_real_$(date +%H%M%S).log 2>&1 < /dev/null &
sleep 25
read MX MY YAW <<< "$(python3 -c "import json;d=json.load(open('/root/jackal_logs/last_goal.json'));print(d['mx'],d['my'],d['yaw'])")"
QZ=$(python3 -c "import math;print(math.sin($YAW/2))"); QW=$(python3 -c "import math;print(math.cos($YAW/2))")
timeout 15 ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped "{header: {frame_id: map}, pose: {position: {x: $MX, y: $MY, z: 0.0}, orientation: {z: $QZ, w: $QW}}}" >/dev/null 2>&1
sleep 12
echo "tracker human_scan_mask_radius_m = $(timeout 8 ros2 param get /safe_path_tracker human_scan_mask_radius_m 2>&1 | tail -1)  hard_stop_mask = $(timeout 8 ros2 param get /safe_path_tracker human_scan_mask_hard_stop 2>&1 | tail -1)"
echo "robot cmd_vel publishers: $(timeout 6 ros2 topic info /j100_0519/cmd_vel 2>&1 | grep 'Publisher count')  (motion OFF)"
echo "bridge: $(tail -4 $(ls -t $R/spubert_real_*.log | head -1) | grep -E 'hold|path_valid' | tail -1 | cut -c60-160)"
python3 /root/pre.py 2>&1 | grep -v XMLPARSER
