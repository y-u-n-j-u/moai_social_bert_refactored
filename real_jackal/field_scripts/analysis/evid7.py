import rclpy, json, time, math, collections, numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid
rclpy.init(); n=Node("evid7"); ev=[]; st={}
q=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL)
n.create_subscription(String,"/spu_bert/selection_evidence",lambda m:ev.append(json.loads(m.data)),100)
n.create_subscription(OccupancyGrid,"/spu_bert/debug_rolling_map",lambda m:st.__setitem__("map",m),q)
t0=time.time()
while time.time()-t0<12: rclpy.spin_once(n,timeout_sec=0.05)
cur=0; tot=0; first=[]; reasons=collections.Counter(); allinv=[]
for d in ev:
    c=d["candidates"]
    ok=any(x["valid"] for x in c)
    for x in c:
        col=x.get("collision") or {}
        if not x["valid"]:
            reasons[x["reason"]]+=1
            tot+=1; cur+= bool(col.get("current_footprint_collision"))
            if col.get("first_collision_distance_m") is not None: first.append(col["first_collision_distance_m"])
print("cycles %d; invalid candidates %d"%(len(ev),tot))
print("invalid because the robot's OWN current footprint already collides (current_footprint_collision): %d of %d"%(cur,tot))
print("distance along the path to the first collision (m): median %.2f  min %.2f  max %.2f  (n=%d)"%((np.median(first),min(first),max(first),len(first)) if first else (0,0,0,0)))
print("reasons:",dict(reasons.most_common()))
d=ev[-1]; px,py=d["robot_pose"][0],d["robot_pose"][1]; m=st["map"]
g=np.array(m.data,dtype=np.int8).reshape(m.info.height,m.info.width); res=m.info.resolution; ox,oy=m.info.origin.position.x,m.info.origin.position.y
ii,jj=np.mgrid[0:g.shape[0],0:g.shape[1]]; X=ox+(jj+0.5)*res; Y=oy+(ii+0.5)*res; D=np.hypot(X-px,Y-py)
occ=(g==100); unk=(g==-1)
print("robot at (%.2f,%.2f): nearest occupied %.2f m; occupied within 0.34/0.44/0.60 m: %d/%d/%d; unknown within 0.44: %d"%(px,py,D[occ].min() if occ.any() else 9,int((occ&(D<0.34)).sum()),int((occ&(D<0.44)).sum()),int((occ&(D<0.60)).sum()),int((unk&(D<0.44)).sum())))
# lateral clearance left/right within 1.5 m ahead along the robot heading
yaw=d["robot_pose"][2]; u=(X-px)*math.cos(yaw)+(Y-py)*math.sin(yaw); v=-(X-px)*math.sin(yaw)+(Y-py)*math.cos(yaw)
fw=(u>0)&(u<1.5)&occ
L=v[fw&(v>0)]; R=-v[fw&(v<0)]
print("lateral clearance within 1.5 m ahead: left %.2f m, right %.2f m -> free width %.2f m (body diameter with margin = 0.88 m)"%(L.min() if L.size else 9,R.min() if R.size else 9,(L.min() if L.size else 9)+(R.min() if R.size else 9)))
