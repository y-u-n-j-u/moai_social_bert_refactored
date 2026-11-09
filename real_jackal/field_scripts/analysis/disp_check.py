import rclpy, time, json, collections
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import String
rclpy.init(); n=Node("dispchk"); mk=[]; ev=[]
q=QoSProfile(depth=50,reliability=ReliabilityPolicy.RELIABLE)
n.create_subscription(MarkerArray,"/spu_bert/selection_markers",lambda m:mk.append((time.time(),m)),q)
n.create_subscription(String,"/spu_bert/selection_evidence",lambda m:ev.append(json.loads(m.data)),q)
T=time.time()
while time.time()-T<40: rclpy.spin_once(n,timeout_sec=0.05)
sig=[];red=[]
for t,m in mk:
    L=[k for k in m.markers if k.ns=="model_candidates"]
    sig.append(hash(tuple(sorted((k.id,round(k.points[-1].x,1),round(k.points[-1].y,1)) for k in L if k.points))))
    red.append(sum(1 for k in L if k.color.r>0.9 and k.color.g<0.3))
ch=1+sum(1 for i in range(1,len(sig)) if sig[i]!=sig[i-1])
nosel=sum(1 for d in ev if d.get("selected_candidate_index") is None)
print("40 s: %d marker messages, %d evidence cycles (%d of them WITHOUT a selected candidate)"%(len(mk),len(ev),nosel))
print("candidate set redrawn %d times (one per %.1f s)"%(ch,40/ch))
print("RED lines per message: max %d (limit 2)"%max(red))
print("excluded-by-heading (valid but not eligible) rows seen in the evidence: %d"%sum(1 for d in ev for c in d["candidates"] if c["valid"] and c.get("score") and not c["score"]["eligible"]))
