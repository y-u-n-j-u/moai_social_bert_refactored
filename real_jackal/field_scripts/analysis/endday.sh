#!/usr/bin/env bash
source /root/jackal_runtime/scripts/jackal_ros_env.bash >/dev/null 2>&1; source /root/moai_stability_ws/install/setup.bash
R=$(readlink -f /root/jackal_logs/drive_latest)
timeout 20 ros2 service call /spu_bert/enable_motion std_srvs/srv/SetBool "{data: false}" 2>&1 | grep -o "success=[A-Za-z]*.*" | cut -c1-80
for p in $(ps -eo pid,args | awk '/[m]onitor_run.bash/ && !/endday/{print $1}'); do kill -TERM $p; done
for p in $(ps -eo pid,args | awk '/[b]ag record/ && !/endday/{print $1}'); do kill -INT $p; done; sleep 8
echo "recorder alive: $(ps -eo args | grep -c '[b]ag record')  monitor alive: $(ps -eo args | grep -c '[m]onitor_run.bash')"
ls $R/bag_110655 | tr '\n' ' '; echo
echo "cmd_vel now: $(timeout 4 ros2 topic echo /j100_0519/cmd_vel --once 2>&1 | grep -E 'x:' | head -1)"
grep "$(date +%F)" /root/jackal_logs/stop_reasons.log | tail -3 | cut -c1-130
python3 /root/drive_an.py $R/bag_110655 $(ls -t $R/spubert_real_*.log | head -1) 2>&1 | grep -E "motion window|distance|stops|rotation|people during"
