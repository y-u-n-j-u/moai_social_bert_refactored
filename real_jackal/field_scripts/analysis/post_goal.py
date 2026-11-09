import sys, math, json, collections
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
r=SequentialReader(); r.open(StorageOptions(uri=sys.argv[1],storage_id="sqlite3"),ConverterOptions("cdr","cdr"))
ty={t.name:t.type for t in r.get_all_topics_and_types()}; D=collections.defaultdict(list); t0=None
keep=("/j100_0519/cmd_vel","/aft_mapped_to_init","/spu_bert/tracker_status","/spu_bert/selection_evidence","/spu_bert/final_goal_odom")
while r.has_next():
    tp,d,t=r.read_next()
    if t0 is None: t0=t*1e-9
    if tp in keep: D[tp].append((t*1e-9-t0,deserialize_message(d,get_message(ty[tp]))))
od=[(s,m.pose.pose.position.x,m.pose.pose.position.y) for s,m in D["/aft_mapped_to_init"]]
P=lambda s:min(od,key=lambda x:abs(x[0]-s))
print("tracker status classes over time (first second of each change), t>=85:")
prev=None
for s,m in D["/spu_bert/tracker_status"]:
    k="tracking" if m.data.startswith("tracking") else m.data.split(":")[0]
    if s>=85 and k!=prev: print("  t=%5.1f odom(%5.1f,%5.1f) %s"%(s,P(s)[1],P(s)[2],m.data[:60])); 
    prev=k
print("\nrobot x by 5 s, with commanded v (mean) and the tracker's goal-reached flag:")
for a in range(85,150,5):
    c=[m.linear.x for t,m in D["/j100_0519/cmd_vel"] if a<=t<a+5]; print("  t=%3d-%3d odom x %.1f  cmd v mean %.2f"%(a,a+5,P(a+2.5)[1],sum(c)/len(c) if c else 0))
fg=D["/spu_bert/final_goal_odom"]; print("\nfinal goal (odom) at the end: (%.2f,%.2f)"%(fg[-1][1].pose.position.x,fg[-1][1].pose.position.y))
