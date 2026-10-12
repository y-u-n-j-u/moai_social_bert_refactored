#!/usr/bin/env bash
# Run inside the container (docker exec -i ... bash -s < check_ready.bash): everything worth looking at before a drive.
# Read-only for the robot: it never starts a drive, it only settles AMCL and reports.
source /root/jackal_runtime/scripts/jackal_ros_env.bash >/dev/null
source /root/moai_stability_ws/install/setup.bash
echo "settling AMCL (8 no-motion updates)..."
for i in $(seq 1 8); do timeout 10 ros2 service call /request_nomotion_update std_srvs/srv/Empty >/dev/null 2>&1; sleep 1; done
timeout 8 ros2 topic echo /amcl_pose --once --field pose.covariance 2>&1 | python3 -c '
import sys, re, math
v = [float(x) for x in re.findall(r"-?[0-9.]+(?:e-?[0-9]+)?", sys.stdin.read())]
if len(v) < 36:
    print("AMCL: no pose received - set 2D Pose Estimate"); sys.exit()
sx, sy, sw = math.sqrt(v[0]), math.sqrt(v[7]), math.degrees(math.sqrt(v[35]))
ok = sx <= 0.6 and sy <= 0.2 and sw <= 10
print("AMCL std  x=%.3f m  y=%.3f m  yaw=%.1f deg  ->  %s (need x<=0.60 y<=0.20 yaw<=10)" % (sx, sy, sw, "PASS" if ok else "FAIL - redo 2D Pose Estimate and move 1-2 m by RC"))'
echo "motion: $(timeout 20 ros2 service call /spu_bert/enable_motion std_srvs/srv/SetBool "{data: false}" 2>&1 | grep -o "Motion [a-z]*")"
echo "robot link: $(timeout 150 /root/jackal_runtime/scripts/verify_jackal_link.bash ready 2>&1 | grep -E "^\[FAIL|PASSED|FAILED" | tr "\n" " ")"
for t in /aft_mapped_to_init /scan; do echo "$t: $(timeout 8 ros2 topic hz $t --window 10 2>&1 | grep -m1 "average rate" || echo NO-DATA)"; done
R=$(readlink -f /root/jackal_logs/drive_latest)
echo "fast-livo: $(tail -1 $R/fastlivo_rss.log)   load: $(uptime | sed "s/.*load average: //")"
L=$(ls -t $R/spubert_real_*.log 2>/dev/null | head -1)
echo "bridge: $(grep -E "path_valid|hold:|new_goal" $L 2>/dev/null | tail -1 | sed -E "s/.*\]: //" | cut -c1-80)"
echo "rviz open: $(ps -eo stat,cmd | grep "rviz[2]" | grep -vc "^Z")  (close it before the run if the computer lags)"
