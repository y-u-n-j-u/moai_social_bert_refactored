import rclpy, math, time
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from tf2_ros import Buffer, TransformListener
rclpy.init(); n=Node("pre"); buf=Buffer(); TransformListener(buf,n); st={}
n.create_subscription(PoseStamped,"/spu_bert/final_goal_odom",lambda m:st.__setitem__("g",m),10)
n.create_subscription(PoseStamped,"/goal_pose",lambda m:st.__setitem__("gp",m),10)
n.create_subscription(Path,"/spu_bert/predicted_path",lambda m:st.__setitem__("p",m),10)
n.create_subscription(Path,"/spu_bert/global_path_odom",lambda m:st.__setitem__("gl",m),10)
t0=time.time()
while time.time()-t0<6: rclpy.spin_once(n,timeout_sec=0.1)
try:
    t=buf.lookup_transform("map","odom",rclpy.time.Time()); print("map->odom (%.2f, %.2f)"%(t.transform.translation.x,t.transform.translation.y))
    b=buf.lookup_transform("odom","base_link",rclpy.time.Time()); bx,by=b.transform.translation.x,b.transform.translation.y
except Exception as e: print("tf error",e); bx=by=0
g=st.get("g")
if g: print("goal (odom) (%.2f, %.2f): %.1f m from the robot"%(g.pose.position.x,g.pose.position.y,math.hypot(g.pose.position.x-bx,g.pose.position.y-by)))
else: print("NO GOAL")
gl=st.get("gl")
if gl and gl.poses: print("global path: %d poses, length ~%.1f m"%(len(gl.poses),sum(math.dist((gl.poses[i].pose.position.x,gl.poses[i].pose.position.y),(gl.poses[i+1].pose.position.x,gl.poses[i+1].pose.position.y)) for i in range(len(gl.poses)-1))))
p=st.get("p")
if p and len(p.poses)>1: print("model path: %d points, first (%.2f,%.2f) last (%.2f,%.2f)"%(len(p.poses),p.poses[0].pose.position.x,p.poses[0].pose.position.y,p.poses[-1].pose.position.x,p.poses[-1].pose.position.y))
