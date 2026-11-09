import rclpy, json, time, math, collections, statistics as st
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from nav_msgs.msg import Odometry
rclpy.init(); n=Node("evid6"); ev=[]; od=[]
n.create_subscription(String,"/spu_bert/selection_evidence",lambda m:ev.append((time.time(),json.loads(m.data))),100)
n.create_subscription(Odometry,"/aft_mapped_to_init",lambda m:od.append((time.time(),m.pose.pose.position.x,m.pose.pose.position.y)),qos_profile_sensor_data)
T0=time.time()
while time.time()-T0<30: rclpy.spin_once(n,timeout_sec=0.05)
def dev(c,pose):
    P=c["path"]
    if not P: return None
    p=P[min(3,len(P)-1)]; return abs((math.atan2(p[1]-pose[1],p[0]-pose[0])-pose[2]+math.pi)%(2*math.pi)-math.pi)*180/math.pi
print("  t(s)  speed(m/s)  cycles  valid/invalid  all-invalid  invalid>90deg  sources")
rows=[]
for s in range(0,30,5):
    o=[x for x in od if T0+s<=x[0]<T0+s+5]; e=[d for t,d in ev if T0+s<=t<T0+s+5]
    sp=sum(math.dist(o[i][1:],o[i+1][1:]) for i in range(len(o)-1))/5 if len(o)>1 else 0
    v=i_=back=0; ai=0
    for d in e:
        c=d["candidates"]
        if c and not any(x["valid"] for x in c): ai+=1
        for x in c:
            a=dev(x,d["robot_pose"])
            if a is None: continue
            if x["valid"]: v+=1
            else:
                i_+=1; back+= a>90
    src=dict(collections.Counter(d["executed_source"] for d in e))
    rows.append((sp,v,i_,ai,len(e)))
    print("  %2d-%2d   %.2f      %2d      %2d/%2d        %d/%d          %d/%d       %s"%(s,s+5,sp,len(e),v,i_,ai,len(e),back,i_,src))
mv=[r for r in rows if r[0]>0.08]; stp=[r for r in rows if r[0]<=0.08]
def tot(L): 
    v=sum(r[1] for r in L); i=sum(r[2] for r in L); a=sum(r[3] for r in L); c=sum(r[4] for r in L)
    return "invalid %.0f%% of candidates, all-invalid cycles %.0f%%"%(100*i/max(v+i,1),100*a/max(c,1)) if c else "no segments"
print("\nmoving (>0.08 m/s):",tot(mv)); print("standing (<=0.08 m/s):",tot(stp))
