import sys, math, collections
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
B=sys.argv[1]; r=SequentialReader(); r.open(StorageOptions(uri=B,storage_id="sqlite3"),ConverterOptions("cdr","cdr"))
ty={t.name:t.type for t in r.get_all_topics_and_types()}; T=collections.defaultdict(list); MO=[]; cmd=[]
while r.has_next():
    tp,d,t=r.read_next(); t*=1e-9
    if tp in ("/aft_mapped_to_init","/scan","/spu_bert/predicted_path"): T[tp].append(t)
    elif tp=="/j100_0519/cmd_vel":
        m=deserialize_message(d,get_message(ty[tp])); cmd.append((t,m.linear.x))
    elif tp=="/tf":
        for tr in deserialize_message(d,get_message(ty[tp])).transforms:
            if tr.header.frame_id=="map" and tr.child_frame_id=="odom":
                q=tr.transform.rotation; MO.append((t,tr.transform.translation.x,tr.transform.translation.y,math.degrees(math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y**2+q.z**2)))))
ts=[t for t,v in cmd if abs(v)>0.01][0]
for name in ("/aft_mapped_to_init","/scan","/spu_bert/predicted_path"):
    L=T[name]; g=[(L[i]-ts,L[i+1]-L[i]) for i in range(len(L)-1) if L[i+1]-L[i]>(0.35 if name!="/spu_bert/predicted_path" else 1.5) and L[i]>=ts-1]
    print("%-28s gaps >%s s inside the drive: %d  %s"%(name,"0.35" if name!="/spu_bert/predicted_path" else "1.5",len(g),[("t=%.1f:%.1fs"%x) for x in g[:10]]))
print("map->odom during the drive (every ~10 s):")
last=-99
for t,x,y,yaw in MO:
    if t-ts>=last+10 and t>=ts-1 and t-ts<=75: last=t-ts; print("   t=%5.1f (%.3f, %.3f) yaw %.1f"%(t-ts,x,y,yaw))
xs=[m[1] for m in MO if ts<=m[0]<=ts+75]; ys=[m[2] for m in MO if ts<=m[0]<=ts+75]
print("map->odom range x %.2f m  y %.2f m  (steps >0.15 m: %d)"%(max(xs)-min(xs),max(ys)-min(ys),sum(1 for a,b in zip(MO,MO[1:]) if ts<=a[0]<=ts+75 and math.hypot(b[1]-a[1],b[2]-a[2])>0.15)))
