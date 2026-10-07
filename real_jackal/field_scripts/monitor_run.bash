#!/usr/bin/env bash
# Gate + arm + supervised drive. Auto-disarm on: FAST-LIVO memory, odom blow-up, localization jump, bridge error, contact range.
source /root/jackal_runtime/scripts/jackal_ros_env.bash >/dev/null
source /root/moai_stability_ws/install/setup.bash
R=$(readlink -f /root/jackal_logs/fallback_test_latest); L=$(ls -t $R/spubert_real_*.log | head -1)
disarm() { timeout 20 ros2 service call /spu_bert/enable_motion std_srvs/srv/SetBool "{data: false}" 2>&1 | grep -o "success=[A-Za-z]*.*"; }
state() { grep bridge $L | grep -E "path_valid|hold:|recovery_sweep" | tail -1 | sed -E "s/.*\]: //" | cut -c1-120; }
mo() { timeout 3 ros2 run tf2_ros tf2_echo map odom 2>&1 | grep -m1 Translation | grep -o "\[.*\]" | tr -d "[],"; }
amcl() { timeout 8 ros2 topic echo /amcl_pose --once 2>&1 | python3 -c '
import sys,re,math
t=sys.stdin.read(); m=re.search(r"covariance:(.*)",t,re.S)
if not m: print("na na na"); sys.exit()
v=[float(x) for x in re.findall(r"-?\d+\.?\d*(?:e-?\d+)?",m.group(1))][:36]
print("%.3f %.3f %.1f"%(math.sqrt(v[0]),math.sqrt(v[7]),math.degrees(math.sqrt(v[35]))))'; }
read SX SY SYAW <<< "$(amcl)"; echo "AMCL std x=$SX y=$SY yaw=${SYAW}deg"
if [ "$SX" = "na" ] || awk -v a="$SX" -v b="$SY" -v c="$SYAW" 'BEGIN{exit !(a>0.60||b>0.20||c>10)}'; then echo "NOT ARMING - AMCL not converged (need x std <=0.60 m, y std <=0.20 m, yaw <=10 deg): re-do 2D Pose Estimate"; exit 3; fi
for i in $(seq 1 10); do case "$(state)" in *waiting_for_goal*|"") sleep 2;; *) break;; esac; done
case "$(state)" in path_valid*) ;; *) echo "NOT ARMING - latest bridge state: $(state)"; exit 3;; esac
bag_alive() { ps -eo cmd | grep -c "ros2 bag reco[r]d"; }
if [ "$(bag_alive)" -lt 1 ]; then echo "rosbag not running - starting it"; /root/start_bag3.bash; sleep 8; fi
[ "$(bag_alive)" -ge 1 ] || { echo "NOT ARMING - rosbag recorder is not running (no evidence would be kept)"; exit 3; }
echo "rosbag recording: $(ls -dt $R/bag_[0-9]* | grep -v '\.log' | head -1)"
MX0=""; for k in 1 2 3 4 5 6; do read MX0 MY0 MZ0 <<< "$(mo)"; [ -n "$MX0" ] && break; sleep 1; done; echo "map->odom at arm: ${MX0:-UNAVAILABLE} $MY0"; echo "state before arm: $(state)"
echo "ARM: $(timeout 20 ros2 service call /spu_bert/enable_motion std_srvs/srv/SetBool '{data: true}' 2>&1 | grep -o 'success=[A-Za-z]*.*')"
REACHED=0
for i in $(seq 1 240); do
  sleep 1
  TS=$(timeout 3 ros2 topic echo /spu_bert/tracker_status --once --field data 2>&1 | head -1 | cut -c1-48)
  RSS=$(tail -1 $R/fastlivo_rss.log 2>/dev/null | grep -o "rss_mb=[0-9]*\|cpu_now=[0-9]*" | tr "\n" " ")
  OD=$(timeout 3 ros2 topic echo /aft_mapped_to_init --once --field pose.pose.position 2>&1 | grep -E "^x:|^y:" | tr "\n" " " | sed -E "s/x: (-?[0-9.]+) y: (-?[0-9.]+).*/\1 \2/" | awk '{printf "%.1f,%.1f", $1, $2}')
  read MX MY MZ <<< "$(mo)"
  DM=$(awk -v a="$MX" -v b="$MY" -v c="$MX0" -v d="$MY0" 'BEGIN{ if (a=="" || c=="") print "na"; else printf "%.2f", sqrt((a-c)^2+(b-d)^2)}')
  [ "$(bag_alive)" -ge 1 ] || echo "WARNING: rosbag recorder died"
  echo "t+${i}s | $TS | $(state | cut -c1-72) | $RSS| odom=$OD | map-odom shift=${DM}m"
  r=$(echo "$RSS" | grep -o "rss_mb=[0-9]*" | grep -o "[0-9]*"); [ -n "$r" ] && [ "$r" -gt 2000 ] && { echo "FASTLIVO MEMORY -> DISARM"; disarm; break; }
  [ -n "$OD" ] && echo "$OD" | tr "," " " | awk 'NF>=2 {exit !($1+0>80||$1+0<-80||$2+0>80||$2+0<-80)} NF<2 {exit 1}' && { echo "ODOM DIVERGED -> DISARM"; disarm; break; }
  [ "$DM" != "na" ] && awk -v d="$DM" 'BEGIN{exit !(d>1.0)}' && { echo "LOCALIZATION JUMP >1 m -> DISARM"; disarm; break; }
  [ "$(grep -c -E '\[ERROR\]|Traceback' $L)" -gt 0 ] && { echo "BRIDGE ERROR -> DISARM"; disarm; break; }
  d=$(echo "$TS" | grep -o "obstacle_too_close:[0-9.]*" | grep -o "[0-9.]*$"); [ -n "$d" ] && awk -v d="$d" 'BEGIN{exit !(d<0.26)}' && { echo "OBSTACLE <0.26m -> DISARM"; disarm; break; }
  case "$TS|$(state)" in *goal_reached*) echo "GOAL REACHED"; REACHED=1; break;; esac
done
# never leave the robot armed without a supervisor: any exit that is not a goal arrival disarms
[ "$REACHED" = 1 ] || { echo "MONITOR ENDED -> DISARM"; disarm; }
