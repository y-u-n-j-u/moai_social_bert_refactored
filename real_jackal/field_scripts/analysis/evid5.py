import rclpy, json, time, math, collections, statistics as st
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from nav_msgs.msg import Odometry
rclpy.init(); n=Node("evid5"); ev=[]; od=[]
n.create_subscription(String,"/spu_bert/selection_evidence",lambda m:ev.append((time.time(),json.loads(m.data))),50)
n.create_subscription(Odometry,"/aft_mapped_to_init",lambda m:od.append((time.time(),m.pose.pose.position.x,m.pose.pose.position.y)),qos_profile_sensor_data)
t0=time.time()
while time.time()-t0<20: rclpy.spin_once(n,timeout_sec=0.05)
dist=sum(math.dist(od[i][1:],od[i+1][1:]) for i in range(len(od)-1)); dur=od[-1][0]-od[0][0]
print("robot moved %.2f m in %.0f s (mean %.2f m/s)"%(dist,dur,dist/max(dur,1e-3)))
def dev(c,pose):
    P=c["path"]
    if not P: return None
    p=P[min(3,len(P)-1)]; return abs((math.atan2(p[1]-pose[1],p[0]-pose[0])-pose[2]+math.pi)%(2*math.pi)-math.pi)*180/math.pi
V=[];I=[];reasons=collections.Counter();allinv=0
for _,d in ev:
    c=d["candidates"]
    if c and not any(x["valid"] for x in c): allinv+=1
    for x in c:
        a=dev(x,d["robot_pose"])
        if a is None: continue
        (V if x["valid"] else I).append(a)
        if not x["valid"]: reasons[x["reason"]]+=1
print("cycles %d | candidates valid %d invalid %d (%.0f%% invalid)"%(len(ev),len(V),len(I),100*len(I)/max(len(V)+len(I),1)))
print("cycles with ALL candidates invalid: %d (%.0f%%)"%(allinv,100*allinv/max(len(ev),1)))
f=lambda L:"median %.0f deg, >90deg %.0f%%"%(st.median(L),100*sum(x>90 for x in L)/len(L)) if L else "none"
print("start-direction deviation  valid: %s | invalid: %s"%(f(V),f(I)))
print("invalid reasons:",dict(reasons.most_common(6)))
print("executed_source:",dict(collections.Counter(d["executed_source"] for _,d in ev)))
d=ev[-1][1]; print("collision_context footprint radius:",d["collision_context"].get("footprint_radius_m"),"robot radius",d["collision_context"].get("robot_radius_m"))
