import sys, math, collections, re
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
B=sys.argv[1]; LOG=sys.argv[2]; r=SequentialReader(); r.open(StorageOptions(uri=B,storage_id="sqlite3"),ConverterOptions("cdr","cdr"))
ty={t.name:t.type for t in r.get_all_topics_and_types()}; ev=collections.defaultdict(list)
while r.has_next():
    tp,d,t=r.read_next()
    if tp in ("/j100_0519/cmd_vel","/spu_bert/tracker_status","/aft_mapped_to_init"): ev[tp].append((t*1e-9,deserialize_message(d,get_message(ty[tp]))))
cmd=ev["/j100_0519/cmd_vel"]; ts=[t for t,m in cmd if abs(m.linear.x)>0.01][0]
P=[(t,m.pose.pose.position.x,m.pose.pose.position.y) for t,m in ev["/aft_mapped_to_init"]]
def pos(t): return min(P,key=lambda p:abs(p[0]-t))[1:]
print("--- tracker_status changes (class), with odom position; x decreasing = moved backwards (teleop)")
prev=None; last_x=None
for t,m in ev["/spu_bert/tracker_status"]:
    s=m.data; k="tracking" if s.startswith("tracking") else s.split(":")[0]
    x,y=pos(t)
    if k!=prev: print("  t=%5.1f odom(%5.1f,%5.1f) %-28s %s"%(t-ts,x,y,k,s[:70] if k!="tracking" else s[:60])); prev=k
print("--- bridge reasons (relative time) when the tracker had no path:")
ev_b=[]
for l in open(LOG):
    m=re.search(r"\[(\d+\.\d+)\] \[real_jackal_spubert_bridge\]: (hold:\S+|path_valid global_fallback.*trigger=\S+)",l)
    if m and ts-1<=float(m.group(1))<=ts+80:
        s=m.group(2); s=re.sub(r"points=\S+ shift=\S+ curvature=\S+ heading=\S+ ","",s); ev_b.append((float(m.group(1))-ts,s[:140]))
last=None
for t,s in ev_b:
    if s!=last: print("  t=%5.1f %s"%(t,s)); last=s
