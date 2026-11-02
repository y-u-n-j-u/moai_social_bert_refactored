import sys, math, collections
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
B=sys.argv[1]; r=SequentialReader(); r.open(StorageOptions(uri=B,storage_id="sqlite3"),ConverterOptions("cdr","cdr"))
ty={t.name:t.type for t in r.get_all_topics_and_types()}; ev=collections.defaultdict(list)
want=("/j100_0519/cmd_vel","/spu_bert/tracker_status","/ped_tracking","/scan","/spu_bert/scan_nohuman","/aft_mapped_to_init")
while r.has_next():
    tp,d,t=r.read_next()
    if tp in want: ev[tp].append((t*1e-9,deserialize_message(d,get_message(ty[tp]))))
cmd=ev["/j100_0519/cmd_vel"]; ts=[t for t,m in cmd if abs(m.linear.x)>0.01][0]; te=[t for t,m in cmd if abs(m.linear.x)>0.01][-1]
print("motion %.1f s; last status lines:"%(te-ts))
for t,m in ev["/spu_bert/tracker_status"]:
    if te-ts-3.5<=t-ts<=te-ts+1.2: print("  t=%5.1f %s"%(t-ts,m.data[:95]))
def near(sc,t):
    best=min(ev[sc],key=lambda x:abs(x[0]-t))[1]; pts=[]
    for i,v in enumerate(best.ranges):
        if math.isfinite(v) and best.range_min<=v<0.8:
            a=best.angle_min+i*best.angle_increment; pts.append((v,math.degrees(a)))
    return sorted(pts)[:4]
for t in (te-2.0,te-1.0,te-0.3,te):
    tr=min(ev["/ped_tracking"],key=lambda x:abs(x[0]-t))[1]
    pe=[(round(math.hypot(k.pose.position.x,k.pose.position.y),2),round(math.degrees(math.atan2(k.pose.position.y,k.pose.position.x)))) for k in tr.tracks]
    print("t=%5.1f closest raw scan points (range m, bearing deg) %s | people tracked (dist, bearing) %s | nohuman nearest %s"%(t-ts,[(round(v,2),round(a)) for v,a in near("/scan",t)],pe,[(round(v,2),round(a)) for v,a in near("/spu_bert/scan_nohuman",t)][:2]))
