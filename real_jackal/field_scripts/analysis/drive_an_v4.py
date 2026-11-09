import sys, math, json, collections, numpy as np
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
B=sys.argv[1]
r=SequentialReader(); r.open(StorageOptions(uri=B,storage_id="sqlite3"),ConverterOptions("cdr","cdr"))
ty={t.name:t.type for t in r.get_all_topics_and_types()}; D=collections.defaultdict(list)
keep=("/aft_mapped_to_init","/spu_bert/selection_evidence","/spu_bert/debug_rolling_map","/spu_bert/selection_markers","/tf")
while r.has_next():
    tp,d,t=r.read_next()
    if tp in keep:
        m=deserialize_message(d,get_message(ty[tp]))
        if tp=="/tf":
            for x in m.transforms:
                if x.header.frame_id=="map" and x.child_frame_id=="odom": D["mo"].append((t*1e-9,x.transform.translation.x))
        else: D[tp].append((t*1e-9,m))
t0=D["/aft_mapped_to_init"][0][0]; mo=D["mo"][0][1]
print("map->odom x = %.2f  (so odom x of the hold 2.4 = map x %.1f)"%(mo,mo+2.4))
ev=[(t-t0,json.loads(m.data)) for t,m in D["/spu_bert/selection_evidence"]]
maps=[(t-t0,m) for t,m in D["/spu_bert/debug_rolling_map"]]
for T in (170,175,185,189):
    s,m=min(maps,key=lambda x:abs(x[0]-T)); e=min(ev,key=lambda x:abs(x[0]-s))[1]; px,py,yaw=e["robot_pose"]
    g=np.array(m.data,dtype=np.int8).reshape(m.info.height,m.info.width); res=m.info.resolution; ox,oy=m.info.origin.position.x,m.info.origin.position.y
    ii,jj=np.mgrid[0:g.shape[0],0:g.shape[1]]; X=ox+(jj+0.5)*res; Y=oy+(ii+0.5)*res; Dd=np.hypot(X-px,Y-py); occ=g==100
    u=(X-px)*math.cos(yaw)+(Y-py)*math.sin(yaw); v=-(X-px)*math.sin(yaw)+(Y-py)*math.cos(yaw)
    ahead=occ&(u>0)&(u<3.0)&(np.abs(v)<0.6); L=v[occ&(u>0)&(u<1.5)&(v>0)]; Rr=-v[occ&(u>0)&(u<1.5)&(v<0)]
    print("t=%3.0f robot odom (%.1f,%.1f) yaw %.0f deg | nearest occupied %.2f m | occupied cells in the 1.2 m-wide strip 0-3 m ahead: %d (nearest at %.2f m ahead) | lateral clearance left %.2f right %.2f"%(s,px,py,math.degrees(yaw),Dd[occ].min(),int(ahead.sum()),u[ahead].min() if ahead.any() else 9.0,L.min() if L.size else 9,Rr.min() if Rr.size else 9))
# candidate display refresh: autonomous window
sm=[(t-t0,m) for t,m in D["/spu_bert/selection_markers"] if 164<=t-t0<=204]
sig=[hash(tuple(sorted((k.id,round(k.points[-1].x,1),round(k.points[-1].y,1)) for k in m.markers if k.ns=="model_candidates" and k.points))) for t,m in sm]
ch=sum(1 for i in range(1,len(sig)) if sig[i]!=sig[i-1])+1
w=[d for s,d in ev if 164<=s<=204]; sel=sum(1 for d in w if d.get("selected_candidate_index") is not None)
print("\nautonomous 40 s: candidate set redrawn %d times (one per %.1f s); cycles WITH a selected candidate %d of %d (the hold only applies to those)"%(ch,40/ch,sel,len(w)))
