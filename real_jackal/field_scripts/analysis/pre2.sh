#!/usr/bin/env bash
source /root/jackal_runtime/scripts/jackal_ros_env.bash >/dev/null 2>&1; source /root/moai_stability_ws/install/setup.bash
R=$(readlink -f /root/jackal_logs/drive_latest)
for i in 1 2 3 4 5 6; do timeout 10 ros2 service call /request_nomotion_update std_srvs/srv/Empty >/dev/null 2>&1; sleep 1; done
echo "AMCL 'Setting pose' events:"; grep "Setting pose" $R/nav2*.log 2>/dev/null | tail -3 | cut -c40-130; ls $R/ | grep nav2 | head -2
python3 /root/loc.py 2>&1 | grep -v XMLPARSER | grep -E "map->odom|x |AMCL|LIO"
python3 /root/pre.py 2>&1 | grep -v XMLPARSER
echo "bridge: $(tail -4 $(ls -t $R/spubert_real_*.log | head -1) | grep -E 'hold|path_valid' | tail -1 | cut -c60-170)"
echo "people in view: $(timeout 4 ros2 topic echo /ped_tracking --once --field tracks 2>/dev/null | grep -c '^- id')"
