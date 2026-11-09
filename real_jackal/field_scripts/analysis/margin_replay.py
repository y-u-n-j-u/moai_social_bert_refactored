import sys, math, glob, collections, numpy as np
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from moai_jackal_spubert.rolling_laser_map import RollingLaserMapProvider
ROBOT_R=0.34; MARGINS=(0.10,0.05,0.00)
def load(bag):
    r=SequentialReader(); r.open(StorageOptions(uri=bag,storage_id="sqlite3"),ConverterOptions("cdr","cdr"))
    ty={t.name:t.type for t in r.get_all_topics_and_types()}; maps=[];od=[];cmd=[]
    if "/spu_bert/debug_rolling_map" not in ty: return None
    while r.has_next():
        tp,d,t=r.read_next(); t*=1e-9
        if tp=="/spu_bert/debug_rolling_map": maps.append((t,deserialize_message(d,get_message(ty[tp]))))
        elif tp=="/aft_mapped_to_init": m=deserialize_message(d,get_message(ty[tp])); od.append((t,m.pose.pose.position.x,m.pose.pose.position.y))
        elif tp=="/j100_0519/cmd_vel": m=deserialize_message(d,get_message(ty[tp])); cmd.append((t,m.linear.x))
    return maps,od,cmd
def collides(msg,x,y,margin,prov):
    g=np.array(msg.data,dtype=np.int8).reshape(msg.info.height,msg.info.width)
    grid=np.where(g==100,2,np.where(g==-1,0,1)).astype(np.uint8)
    prov.grid=grid; prov.origin_x=msg.info.origin.position.x; prov.origin_y=msg.info.origin.position.y
    return prov.point_occupied_with_radius(x,y,ROBOT_R+margin)
tot={m:[0,0] for m in MARGINS}; stops_all=[]
for bag in sys.argv[1:]:
    data=load(bag)
    if not data: print(bag.split("/")[-1],": no rolling map recorded"); continue
    maps,od,cmd=data
    first=maps[0][1]; prov=RollingLaserMapProvider(size_m=first.info.width*first.info.resolution,resolution=first.info.resolution)
    ts=[t for t,v in cmd if abs(v)>0.01][0]; te=[t for t,v in cmd if abs(v)>0.01][-1]
    pos=lambda t:min(od,key=lambda p:abs(p[0]-t))
    flags={m:[] for m in MARGINS}
    for t,msg in maps:
        if not ts<=t<=te: continue
        p=pos(t)
        for m in MARGINS: flags[m].append((t,collides(msg,p[1],p[2],m,prov)))
    n=len(flags[MARGINS[0]])
    print("\n== %s: %d map snapshots in the %.0f s drive"%(bag.split("/")[-1],n,te-ts))
    for m in MARGINS:
        c=sum(f for _,f in flags[m]); tot[m][0]+=c; tot[m][1]+=n
        eps=sum(1 for i,(t,f) in enumerate(flags[m]) if f and (i==0 or not flags[m][i-1][1]))
        print("   margin %.2f (radius %.2f): robot's own footprint colliding in %d of %d snapshots (%.0f%%), %d separate episodes"%(m,ROBOT_R+m,c,n,100*c/max(n,1),eps))
    # actual stops
    win=[(t,v) for t,v in cmd if ts<=t<=te]; stops=[];st=None
    for t,v in win:
        s=abs(v)<0.02
        if s and st is None: st=t
        if not s and st is not None:
            if t-st>0.5: stops.append((st,t))
            st=None
    for a,b in stops:
        near=min([x for x in flags[0.10] if abs(x[0]-a)<=2.0],key=lambda x:abs(x[0]-a),default=None)
        if not near: continue
        row={m:min([x for x in flags[m] if abs(x[0]-a)<=2.0],key=lambda x:abs(x[0]-a))[1] for m in MARGINS}
        stops_all.append((bag.split("/")[-1],a-ts,b-a,row))
print("\n=== all three bags:")
for m in MARGINS: print("   margin %.2f: own footprint colliding in %.0f%% of %d snapshots"%(m,100*tot[m][0]/max(tot[m][1],1),tot[m][1]))
print("\n=== actual stops (>0.5 s) vs the start-collision flag at that moment (True = every candidate would be rejected):")
print("   bag               t(s)  length  m=0.10  m=0.05  m=0.00")
for b,t,l,row in stops_all: print("   %-16s %5.0f  %4.1fs   %-6s  %-6s  %-6s"%(b,t,l,row[0.10],row[0.05],row[0.00]))
s10=[r for r in stops_all if r[3][0.10]]; print("\nstops with start-collision at margin 0.10: %d of %d; resolved at 0.05: %d; at 0.00: %d"%(len(s10),len(stops_all),sum(1 for r in s10 if not r[3][0.05]),sum(1 for r in s10 if not r[3][0.00])))
print("seconds of those stops: total %.1f s; resolved at 0.05: %.1f s"%(sum(r[2] for r in s10),sum(r[2] for r in s10 if not r[3][0.05])))
