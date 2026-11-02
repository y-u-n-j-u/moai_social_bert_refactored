import rclpy, math, time
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
rclpy.init(); n=Node("loc"); buf=Buffer(); TransformListener(buf,n)
cov=[]; od=[]
n.create_subscription(PoseWithCovarianceStamped,"/amcl_pose",lambda m:cov.append((time.time(),m)),10)
n.create_subscription(Odometry,"/aft_mapped_to_init",lambda m:od.append((time.time(),m.pose.pose.position.x,m.pose.pose.position.y)),qos_profile_sensor_data)
S=[]; t0=time.time()
while time.time()-t0<20:
    rclpy.spin_once(n,timeout_sec=0.05)
    try:
        t=buf.lookup_transform("map","odom",rclpy.time.Time()); q=t.transform.rotation
        yaw=math.degrees(math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y**2+q.z**2)))
        S.append((time.time()-t0,t.transform.translation.x,t.transform.translation.y,yaw))
    except Exception: pass
    time.sleep(0.2)
print("map->odom samples:",len(S))
if S:
    xs=[s[1] for s in S]; ys=[s[2] for s in S]; yw=[s[3] for s in S]
    print("  x %.3f..%.3f (range %.3f)  y %.3f..%.3f (range %.3f)  yaw %.1f..%.1f (range %.1f deg)"%(min(xs),max(xs),max(xs)-min(xs),min(ys),max(ys),max(ys)-min(ys),min(yw),max(yw),max(yw)-min(yw)))
    steps=[math.hypot(S[i+1][1]-S[i][1],S[i+1][2]-S[i][2]) for i in range(len(S)-1)]
    print("  largest step between samples %.3f m; steps >0.1 m: %d"%(max(steps),sum(s>0.1 for s in steps)))
if cov:
    v=cov[-1][1].pose.covariance; print("AMCL std now: x %.3f y %.3f yaw %.1f deg; amcl_pose msgs in 20 s: %d"%(math.sqrt(v[0]),math.sqrt(v[7]),math.degrees(math.sqrt(v[35])),len(cov)))
else: print("amcl_pose: no message in 20 s (normal while standing still)")
if len(od)>2:
    gaps=[od[i+1][0]-od[i][0] for i in range(len(od)-1)]; d=math.hypot(od[-1][1]-od[0][1],od[-1][2]-od[0][2])
    print("LIO odom: %d msgs, max gap %.2f s, drift over 20 s while standing: %.3f m"%(len(od),max(gaps),d))
