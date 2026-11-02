import sys, math, collections
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
B=sys.argv[1]; r=SequentialReader(); r.open(StorageOptions(uri=B,storage_id="sqlite3"),ConverterOptions("cdr","cdr"))
ty={t.name:t.type for t in r.get_all_topics_and_types()}; ev=collections.defaultdict(list)
while r.has_next():
    tp,d,t=r.read_next()
    if tp in ("/j100_0519/cmd_vel","/scan","/ped_tracking"): ev[tp].append((t*1e-9,deserialize_message(d,get_message(ty[tp]))))
ts=[t for t,m in ev["/j100_0519/cmd_vel"] if abs(m.linear.x)>0.01][0]
for rel in (54.6,55.0,55.4,55.7,55.9,56.1,56.3):
    t=ts+rel
    sc=min(ev["/scan"],key=lambda x:abs(x[0]-t))[1]; pts=sorted((v,math.degrees(sc.angle_min+i*sc.angle_increment)) for i,v in enumerate(sc.ranges) if math.isfinite(v) and sc.range_min<=v<1.0)[:3]
    tr=min(ev["/ped_tracking"],key=lambda x:abs(x[0]-t))[1]
    print("t=%5.1f nearest scan pts %s | tracks %s"%(rel,[(round(v,2),round(a)) for v,a in pts],[(round(k.pose.position.x,2),round(k.pose.position.y,2),round(k.score,2)) for k in tr.tracks]))
