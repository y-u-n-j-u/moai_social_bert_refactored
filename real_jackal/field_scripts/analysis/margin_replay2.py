import sys, math, numpy as np
sys.argv_saved=sys.argv
exec(open("/root/margin_replay.py").read().split("tot={m:[0,0]")[0])     # reuse load()
def exact(msg,x,y,R):
    g=np.array(msg.data,dtype=np.int8).reshape(msg.info.height,msg.info.width); res=msg.info.resolution
    ii,jj=np.nonzero((g==100)|(g==-1)); gx=msg.info.origin.position.x+(jj+0.5)*res; gy=msg.info.origin.position.y+(ii+0.5)*res
    return bool((np.hypot(gx-x,gy-y)<R).any())
def snapped(msg,x,y,R,prov):
    return collides(msg,x,y,R-ROBOT_R,prov)
cases=[("current code, margin 0.10 (R=0.44, snapped, +half cell)",lambda m,x,y,p:snapped(m,x,y,0.44,p)),
       ("current code, margin 0.05",lambda m,x,y,p:snapped(m,x,y,0.39,p)),
       ("EXACT distance, R=0.44 (no snapping, no half-cell)",lambda m,x,y,p:exact(m,x,y,0.44)),
       ("EXACT distance, R=0.39",lambda m,x,y,p:exact(m,x,y,0.39)),
       ("EXACT distance, R=0.34 (hull only)",lambda m,x,y,p:exact(m,x,y,0.34))]
stops_all=[]; counts={c[0]:[0,0] for c in cases}
for bag in sys.argv_saved[1:]:
    maps,od,cmd=load(bag); first=maps[0][1]; prov=RollingLaserMapProvider(size_m=first.info.width*first.info.resolution,resolution=first.info.resolution)
    ts=[t for t,v in cmd if abs(v)>0.01][0]; te=[t for t,v in cmd if abs(v)>0.01][-1]
    pos=lambda t:min(od,key=lambda p:abs(p[0]-t))
    fl={c[0]:[] for c in cases}
    for t,msg in maps:
        if not ts<=t<=te: continue
        p=pos(t)
        for name,fn in cases: fl[name].append((t,fn(msg,p[1],p[2],prov)))
    for name,_ in cases: counts[name][0]+=sum(f for _,f in fl[name]); counts[name][1]+=len(fl[name])
    win=[(t,v) for t,v in cmd if ts<=t<=te]; st=None
    for t,v in win:
        s=abs(v)<0.02
        if s and st is None: st=t
        if not s and st is not None:
            if t-st>0.5:
                row={name:min([x for x in fl[name] if abs(x[0]-st)<=2.0],key=lambda x:abs(x[0]-st))[1] for name,_ in cases if [x for x in fl[name] if abs(x[0]-st)<=2.0]}
                stops_all.append((bag.split("/")[-1],st-ts,t-st,row))
            st=None
print("share of drive snapshots where the robot's own footprint collides:")
for name,_ in cases: print("   %-58s %4.1f%%"%(name,100*counts[name][0]/max(counts[name][1],1)))
base=cases[0][0]; s10=[r for r in stops_all if r[3].get(base)]
print("\nof the %d stops with a start collision under the current code (total %.1f s), how many are NOT colliding under each variant:"%(len(s10),sum(r[2] for r in s10)))
for name,_ in cases[1:]:
    res=[r for r in s10 if not r[3].get(name,True)]; print("   %-58s %d of %d stops  (%.1f s)"%(name,len(res),len(s10),sum(r[2] for r in res)))
