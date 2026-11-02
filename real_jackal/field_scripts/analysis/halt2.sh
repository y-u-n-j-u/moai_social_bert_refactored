#!/usr/bin/env bash
source /root/jackal_runtime/scripts/jackal_ros_env.bash >/dev/null 2>&1; source /root/moai_stability_ws/install/setup.bash
echo "$(date +%F_%T) $(date +%s) claude: user said stop" >> /root/jackal_logs/stop_reasons.log
timeout 20 ros2 service call /spu_bert/enable_motion std_srvs/srv/SetBool "{data: false}" 2>&1 | grep -o "success=[A-Za-z]*.*" | cut -c1-90
for p in $(ps -eo pid,args | awk '/[m]onitor_run.bash/ && !/halt2/{print $1}'); do kill -TERM $p; done
sleep 2
echo "tracker: $(timeout 5 ros2 topic echo /spu_bert/tracker_status --once --field data 2>&1 | head -1 | cut -c1-70) | cmd_vel: $(timeout 4 ros2 topic echo /j100_0519/cmd_vel --once 2>&1 | grep -E 'x:' | head -1)"
echo "monitor alive: $(ps -eo args | grep -c '[m]onitor_run.bash')  recorder alive: $(ps -eo args | grep -c '[b]ag record')"
