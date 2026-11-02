import sys, math, collections, re
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
B=sys.argv[1]; LOG=sys.argv[2]; r=SequentialReader(); r.open(StorageOptions(uri=B,storage_id="sqlite3"),ConverterOptions("cdr","cdr"))
ty={t.name:t.type for t in r.get_all_topics_and_types()}; ev=collections.defaultdict(list)
want=("/j100_0519/cmd_vel","/spu_bert/tracker_status","/ped_tracking","/scan","/aft_mapped_to_init","/spu_bert/predicted_path")
while r.has_next():
    tp,d,t=r.read_next()
    if tp in want: ev[tp].append((t*1e-9,deserialize_message(d,get_message(ty[tp]))))
cmd=ev["/j100_0519/cmd_vel"]; nz=[t for t,m in cmd if abs(m.linear.x)>0.01]; ts,te=nz[0],nz[-1]
od=ev["/aft_mapped_to_init"]; P=[(t,m.pose.pose.position.x,m.pose.pose.position.y) for t,m in od]
def pos(t): return min(P,key=lambda p:abs(p[0]-t))[1:]
print("motion window %.1f s; travelled to odom (%.1f, %.1f)"%(te-ts,*pos(te)))
# stops (>0.4 s with v<0.02) with status text
stops=[];st=None
for t,m in cmd:
    if t<ts or t>te+2: continue
    s=abs(m.linear.x)<0.02
    if s and st is None: st=t
    if not s and st is not None:
        if t-st>0.4: stops.append((st,t)); 
        st=None
if st is not None: stops.append((st,te+2))
print("stops (>0.4 s):",len(stops))
status=ev["/spu_bert/tracker_status"]
def at(t): return min(status,key=lambda x:abs(x[0]-t))[1].data[:80]
def scan_stats(t):
    sc=min(ev["/scan"],key=lambda x:abs(x[0]-t))[1]; L=[];Rr=[];F=[]
    for i,v in enumerate(sc.ranges):
        if not math.isfinite(v): continue
        a=sc.angle_min+i*sc.angle_increment; x,y=v*math.cos(a),v*math.sin(a)
        if 0<x<1.5:
            if y>0.05: L.append(y)
            elif y<-0.05: Rr.append(-y)
        if 0<x<1.2 and abs(y)<0.35: F.append(v)
    return (round(min(L),2) if L else None, round(min(Rr),2) if Rr else None, round(min(F),2) if F else None)
tr_all=ev["/ped_tracking"]
for a,b in stops:
    p=pos(a); lw,rw,fr=scan_stats(a)
    tr=min(tr_all,key=lambda x:abs(x[0]-a))[1]
    ppl=[(round(math.hypot(k.pose.position.x,k.pose.position.y),2),round(math.degrees(math.atan2(k.pose.position.y,k.pose.position.x)))) for k in tr.tracks]
    print("t=%5.1f..%5.1f (%.1fs) at odom (%.1f,%.1f): status '%s' | gap left %s right %s m (within 1.5 m ahead) => width %s | front min %s | people %s"%(a-ts,b-ts,b-a,p[0],p[1],at(a),lw,rw,round(lw+rw,2) if lw and rw else None,fr,ppl))
# bridge decisions in window
c=collections.Counter()
for l in open(LOG):
    m=re.search(r"\[(\d+\.\d+)\] \[real_jackal_spubert_bridge\]: (path_valid|hold)(.*)",l)
    if not m or not ts-0.5<=float(m.group(1))<=te+1: continue
    s=m.group(2)+m.group(3)
    k="model rank" if "rank=" in s else "retained" if "retained" in s else ("fallback:"+(re.search(r"trigger=(\S+)",s).group(1) if "trigger=" in s else "?")) if "global_fallback" in s else s[:60]
    c[k]+=1
print("bridge decisions:",dict(c.most_common(10)))
