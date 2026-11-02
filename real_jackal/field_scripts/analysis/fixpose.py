import rclpy, math, time
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PoseWithCovarianceStamped
rclpy.init(); n=Node("fixpose"); buf=Buffer(); TransformListener(buf,n)
pub=n.create_publisher(PoseWithCovarianceStamped,"/initialpose",10)
t0=time.time(); ob=None
while time.time()-t0<5 and ob is None:
    rclpy.spin_once(n,timeout_sec=0.1)
    try: ob=buf.lookup_transform("odom","base_link",rclpy.time.Time())
    except Exception: pass
q=ob.transform.rotation; byaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y**2+q.z**2)); bx,by=ob.transform.translation.x,ob.transform.translation.y
mx,my,myaw=8.40,-0.17,math.radians(-10.0)           # map->odom that held through every earlier recording (robot did not move since)
x=mx+math.cos(myaw)*bx-math.sin(myaw)*by; y=my+math.sin(myaw)*bx+math.cos(myaw)*by; yaw=myaw+byaw
m=PoseWithCovarianceStamped(); m.header.frame_id="map"; m.header.stamp=rclpy.time.Time().to_msg()
m.pose.pose.position.x=x; m.pose.pose.position.y=y; m.pose.pose.orientation.z=math.sin(yaw/2); m.pose.pose.orientation.w=math.cos(yaw/2)
m.pose.covariance[0]=0.04; m.pose.covariance[7]=0.04; m.pose.covariance[35]=0.03
print("odom->base_link (%.3f, %.3f, %.1f deg)  =>  robot on the map (%.3f, %.3f, %.1f deg)"%(bx,by,math.degrees(byaw),x,y,math.degrees(yaw)))
for _ in range(3): pub.publish(m); rclpy.spin_once(n,timeout_sec=0.2); time.sleep(0.5)
