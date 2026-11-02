import rclpy, math, time, json
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener
rclpy.init(); n=Node("prep"); buf=Buffer(); TransformListener(buf,n); st={}
n.create_subscription(PoseStamped,"/spu_bert/final_goal_odom",lambda m:st.__setitem__("g",m),10)
t0=time.time()
while time.time()-t0<6: rclpy.spin_once(n,timeout_sec=0.1)
g=st.get("g")
if not g: print("NO GOAL"); raise SystemExit
t=buf.lookup_transform("map","odom",rclpy.time.Time()); q=t.transform.rotation; myaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y**2+q.z**2))
x,y=g.pose.position.x,g.pose.position.y; gq=g.pose.orientation; gyaw=math.atan2(2*(gq.w*gq.z+gq.x*gq.y),1-2*(gq.y**2+gq.z**2))
mx=t.transform.translation.x+math.cos(myaw)*x-math.sin(myaw)*y; my=t.transform.translation.y+math.sin(myaw)*x+math.cos(myaw)*y
b=buf.lookup_transform("odom","base_link",rclpy.time.Time()).transform.translation
print(json.dumps({"mx":mx,"my":my,"yaw":gyaw+myaw,"dist":math.hypot(x-b.x,y-b.y),"map_odom":[t.transform.translation.x,t.transform.translation.y]}))
