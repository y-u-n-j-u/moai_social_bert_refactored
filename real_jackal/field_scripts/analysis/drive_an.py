import sys, math, glob, re, collections
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
B=sys.argv[1]; LOG=sys.argv[2]
r=SequentialReader(); r.open(StorageOptions(uri=B,storage_id="sqlite3"),ConverterOptions("cdr","cdr"))
ty={t.name:t.type for t in r.get_all_topics_and_types()}
ev=collections.defaultdict(list)
while r.has_next():
    tp,d,t=r.read_next(); ev[tp].append((t*1e-9,deserialize_message(d,get_message(ty[tp]))))
print("topics:",{k.split('/')[-1]:len(v) for k,v in ev.items()})
cmd=ev.get("/j100_0519/cmd_vel",[]); od=ev["/aft_mapped_to_init"]
t0=cmd[0][0] if cmd else od[0][0]
moving=[(t,m.linear.x,m.angular.z) for t,m in cmd]
# motion window = first nonzero command to last nonzero command
nz=[x for x in moving if abs(x[1])>0.01 or abs(x[2])>0.01]
ts,te=nz[0][0],nz[-1][0]; dur=te-ts
print("commands: %d msgs; motion window %.1f s (first to last non-zero command)"%(len(cmd),dur))
P=[(t,m.pose.pose.position.x,m.pose.pose.position.y) for t,m in od if ts-1<=t<=te+1]
dist=sum(math.dist(P[i][1:],P[i+1][1:]) for i in range(len(P)-1)); print("distance travelled (LIO odom) %.1f m, mean speed %.2f m/s"%(dist,dist/max(dur,1e-3)))
win=[x for x in moving if ts<=x[0]<=te]
# stops: stretches with |v|<0.02 and |w|<0.05 longer than 0.5 s
stops=[];st=None
for t,v,w in win:
    s=abs(v)<0.02 and abs(w)<0.05
    if s and st is None: st=t
    if not s and st is not None:
        if t-st>0.5: stops.append((st-ts,t-st)); 
        st=None
print("stops >0.5 s: %d, total %.1f s %s"%(len(stops),sum(x[1] for x in stops),[("t=%.0f:%.1fs"%x) for x in stops[:8]]))
rot=sum(abs(w)*(win[i+1][0]-win[i][0]) for i,(t,v,w) in enumerate(win[:-1]))
print("rotation %.0f deg total = %.0f deg per 10 m; |w|>0.4 for %.1f s; max v %.2f, mean v while moving %.2f"%(math.degrees(rot),math.degrees(rot)/max(dist/10,1e-3),sum((win[i+1][0]-win[i][0]) for i,(t,v,w) in enumerate(win[:-1]) if abs(w)>0.4),max(v for _,v,_ in win),sum(v for _,v,_ in win if v>0.02)/max(sum(v>0.02 for _,v,_ in win),1)))
# people
tr=ev.get("/ped_tracking",[])
d=[(t,min(math.hypot(k.pose.position.x,k.pose.position.y) for k in m.tracks)) for t,m in tr if m.tracks and ts<=t<=te]
if d: print("people during the drive: in view %.0f%% of frames; min distance %.2f m at t=%.0f s; ids %s"%(100*len(d)/max(sum(1 for t,_ in tr if ts<=t<=te),1),min(x[1] for x in d),min(d,key=lambda x:x[1])[0]-ts,sorted({int(k.id) for t,m in tr if ts<=t<=te for k in m.tracks})))
else: print("people during the drive: none tracked")
# path sources from bridge log within window (ros time)
c=collections.Counter(); n=0
for l in open(LOG):
    m=re.search(r"\[(\d+\.\d+)\] \[real_jackal_spubert_bridge\]: (path_valid|hold)(.*)",l)
    if not m or not ts-0.5<=float(m.group(1))<=te+0.5: continue
    s=m.group(2)+m.group(3); n+=1
    k="model rank" if "rank=" in s else "retained" if "retained" in s else ("fallback:"+(re.search(r"trigger=(\S+)",s).group(1) if "trigger=" in s else "?")) if "global_fallback" in s else s[:50]
    c[k]+=1
print("bridge path decisions in the window: %d  %s"%(n,dict(c.most_common(8))))
