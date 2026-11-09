import sys, math, json, glob, re, collections, statistics as st, numpy as np
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from moai_jackal_spubert.rolling_laser_map import RollingLaserMapProvider
B=sys.argv[1]; LOG=sys.argv[2]
r=SequentialReader(); r.open(StorageOptions(uri=B,storage_id="sqlite3"),ConverterOptions("cdr","cdr"))
ty={t.name:t.type for t in r.get_all_topics_and_types()}; D=collections.defaultdict(list)
keep=("/j100_0519/cmd_vel","/aft_mapped_to_init","/spu_bert/tracker_status","/spu_bert/selection_evidence","/ped_tracking","/goal_pose","/spu_bert/final_goal_odom","/spu_bert/debug_rolling_map","/spu_bert/selection_markers","/spu_bert/emergency_stop")
while r.has_next():
    tp,d,t=r.read_next()
    if tp in keep: D[tp].append((t*1e-9,deserialize_message(d,get_message(ty[tp]))))
print("topics:",{k.split('/')[-1]:len(v) for k,v in D.items()})
cmd=D["/j100_0519/cmd_vel"]; od=[(t,m.pose.pose.position.x,m.pose.pose.position.y) for t,m in D["/aft_mapped_to_init"]]
t_rec=od[0][0]; t_cmd=[t for t,m in cmd if abs(m.linear.x)>0.01][0]; t_end=od[-1][0]
def speed(t0,t1):
    p=[x for x in od if t0<=x[0]<=t1]
    return math.dist(p[0][1:],p[-1][1:])/max(p[-1][0]-p[0][0],1e-3) if len(p)>1 else 0
bins=[]; T=int(t_end-t_rec)
for s in range(T):
    a=t_rec+s; c=[m for t,m in cmd if a<=t<a+1]; v=sum(m.linear.x for m in c)/len(c) if c else 0
    bins.append((s,v,speed(a,a+1)))
moved=[b for b in bins if b[2]>0.05]; t_move=moved[0][0]; t_stop_move=moved[-1][0]
print("\nrecording %.0f s; first command at %.0f s; robot first moved at %.0f s (%.0f s after the first command); last movement at %.0f s"%(T,t_cmd-t_rec,t_move,t_move-(t_cmd-t_rec),t_stop_move))
win=[b for b in bins if t_move<=b[0]<=t_stop_move]
dist=sum(math.dist(od[i][1:],od[i+1][1:]) for i in range(len(od)-1) if t_rec+t_move<=od[i][0]<=t_rec+t_stop_move+1)
print("moving window %.0f s, travelled %.1f m, mean speed %.2f m/s, max commanded v %.2f"%(len(win),dist,dist/max(len(win),1),max(b[1] for b in bins)))
# episodes inside the window: commanded but not moving / commanded zero
def episodes(pred,minlen=1):
    out=[];st_=None
    for b in win:
        if pred(b) and st_ is None: st_=b[0]
        if (not pred(b)) and st_ is not None:
            if b[0]-st_>=minlen: out.append((st_,b[0]-st_)); 
            st_=None
    return out
hold=episodes(lambda b:b[1]<0.03)
blocked=episodes(lambda b:b[1]>=0.05 and b[2]<0.03)
print("holds inside the moving window (commanded v~0): %d, total %.0f s %s"%(len(hold),sum(l for _,l in hold),[("t=%d:%ds"%x) for x in hold[:12]]))
print("commanded but NOT moving inside the window: %d, total %.0f s %s"%(len(blocked),sum(l for _,l in blocked),[("t=%d:%ds"%x) for x in blocked[:8]]))
pre=[b for b in bins if b[0]<t_move and b[1]>=0.05 and b[2]<0.03]; print("commanded but not moving BEFORE the first movement (e-stop): %d s"%len(pre))
w=[abs(m.angular.z) for t,m in cmd if t_rec+t_move<=t<=t_rec+t_stop_move+1]; dt=1/ (len(w)/max(len(win),1)) if w else 0.1
rot=math.degrees(sum(w)*dt); print("rotation %.0f deg = %.0f deg per 10 m; |w|>0.4 for %.1f s"%(rot,rot/max(dist/10,1e-3),sum(1 for x in w if x>0.4)*dt))
# goals
gp=D["/goal_pose"]; fg=D["/spu_bert/final_goal_odom"]
print("goal_pose messages during the recording: %d at %s"%(len(gp),[round(t-t_rec) for t,_ in gp][:6]))
if fg: print("final goal (odom) first (%.2f,%.2f) last (%.2f,%.2f); robot last odom (%.2f,%.2f)"%(fg[0][1].pose.position.x,fg[0][1].pose.position.y,fg[-1][1].pose.position.x,fg[-1][1].pose.position.y,od[-1][1],od[-1][2]))
# tracker status changes inside the window
print("\ntracker status changes (merged) inside the moving window:")
prev=None
for t,m in D["/spu_bert/tracker_status"]:
    s=t-t_rec
    if not t_move-2<=s<=t_stop_move+3: continue
    k="tracking" if m.data.startswith("tracking") else m.data.split(":")[0]
    if k!=prev: p=min(od,key=lambda x:abs(x[0]-t)); print("  t=%5.0f odom(%5.1f,%5.1f) %s"%(s,p[1],p[2],m.data[:80] if k!="tracking" else "tracking")); prev=k
# people
tr=[(t,m) for t,m in D["/ped_tracking"] if t_rec+t_move<=t<=t_rec+t_stop_move+1]
dmin=[(t-t_rec,min(math.hypot(k.pose.position.x,k.pose.position.y) for k in m.tracks)) for t,m in tr if m.tracks]
print("\npeople during the moving window: in view %.0f%% of frames; min distance %s; ids %s"%(100*len(dmin)/max(len(tr),1),("%.2f m at t=%.0f s"%(min(x[1] for x in dmin),min(dmin,key=lambda x:x[1])[0])) if dmin else "none",sorted({int(k.id) for t,m in tr for k in m.tracks})))
# candidates
ev=[(t,json.loads(m.data)) for t,m in D["/spu_bert/selection_evidence"] if t_rec+t_move<=t<=t_rec+t_stop_move+1]
src=collections.Counter(d["executed_source"] for _,d in ev); n_c=collections.Counter(len(d["candidates"]) for _,d in ev)
V=sum(1 for _,d in ev for c in d["candidates"] if c["valid"]); I=sum(1 for _,d in ev for c in d["candidates"] if not c["valid"])
allinv=sum(1 for _,d in ev if d["candidates"] and not any(c["valid"] for c in d["candidates"]))
reasons=collections.Counter(c["reason"] for _,d in ev for c in d["candidates"] if not c["valid"])
cur=sum(1 for _,d in ev for c in d["candidates"] if not c["valid"] and (c.get("collision") or {}).get("current_footprint_collision"))
print("\nselection cycles in the moving window: %d | executed_source %s"%(len(ev),dict(src)))
print("candidates per cycle: %s | valid %d invalid %d (%.0f%% invalid) | cycles with ALL invalid: %d (%.0f%%)"%(dict(sorted(n_c.items())),V,I,100*I/max(V+I,1),allinv,100*allinv/max(len(ev),1)))
print("invalid reasons: %s | of the invalid, the robot's OWN footprint already collided: %d"%(dict(reasons.most_common(5)),cur))
# start-collision share on rolling map snapshots
maps=[(t,m) for t,m in D["/spu_bert/debug_rolling_map"] if t_rec+t_move<=t<=t_rec+t_stop_move+1]
if maps:
    first=maps[0][1]; prov=RollingLaserMapProvider(size_m=first.info.width*first.info.resolution,resolution=first.info.resolution); col=0
    for t,m in maps:
        p=min(od,key=lambda x:abs(x[0]-t)); g=np.array(m.data,dtype=np.int8).reshape(m.info.height,m.info.width)
        prov.grid=np.where(g==100,2,np.where(g==-1,0,1)).astype(np.uint8); prov.origin_x=m.info.origin.position.x; prov.origin_y=m.info.origin.position.y
        col+=prov.point_occupied_with_radius(p[1],p[2],0.44)
    print("robot's own footprint (r=0.44) colliding in %d of %d rolling-map snapshots (%.0f%%)"%(col,len(maps),100*col/len(maps)))
# candidate display hold: changes of the drawn candidate set
sm=[(t,m) for t,m in D["/spu_bert/selection_markers"] if t_rec+t_move<=t<=t_rec+t_stop_move+1]
sig=[hash(tuple(sorted((k.id,round(k.points[-1].x,1),round(k.points[-1].y,1)) for k in m.markers if k.ns=="model_candidates" and k.points))) for t,m in sm]
ch=1+sum(1 for i in range(1,len(sig)) if sig[i]!=sig[i-1]) if sig else 0
print("drawn candidate set changed %d times in %d s (%d marker messages) => one change per %.1f s"%(ch,len(win),len(sm),len(win)/max(ch,1)))
