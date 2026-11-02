import sys, math, collections, numpy as np
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
B=sys.argv[1]; r=SequentialReader(); r.open(StorageOptions(uri=B,storage_id="sqlite3"),ConverterOptions("cdr","cdr"))
ty={t.name:t.type for t in r.get_all_topics_and_types()}; maps=[]; od=[]; cmd=[]
while r.has_next():
    tp,d,t=r.read_next(); t*=1e-9
    if tp=="/spu_bert/debug_rolling_map": maps.append((t,deserialize_message(d,get_message(ty[tp]))))
    elif tp=="/aft_mapped_to_init": m=deserialize_message(d,get_message(ty[tp])); od.append((t,m.pose.pose.position.x,m.pose.pose.position.y))
    elif tp=="/j100_0519/cmd_vel": m=deserialize_message(d,get_message(ty[tp])); cmd.append((t,m.linear.x))
ts=[t for t,v in cmd if abs(v)>0.01][0]
starts=[float(x) for x in sys.argv[2:]]
print("snapshot of the rolling map (odom frame) at each stop: robot at centre; occupied / unknown cells; free-space check for a body of radius R (R=0.44 is robot 0.34 + margin 0.10)")
for rel in starts:
    t=ts+rel; mt,m=min(maps,key=lambda x:abs(x[0]-t)); p=min(od,key=lambda x:abs(x[0]-t))
    g=np.array(m.data,dtype=np.int8).reshape(m.info.height,m.info.width); res=m.info.resolution; ox,oy=m.info.origin.position.x,m.info.origin.position.y
    ii,jj=np.mgrid[0:g.shape[0],0:g.shape[1]]; X=ox+(jj+0.5)*res; Y=oy+(ii+0.5)*res; D=np.hypot(X-p[1],Y-p[2])
    occ=(g==100); unk=(g==-1)
    near=D[occ].min() if occ.any() else float("inf")
    # direction of the nearest occupied cell in robot-forward-agnostic odom frame
    k=np.argmin(np.where(occ,D,1e9)); dx,dy=X.flat[k]-p[1],Y.flat[k]-p[2]
    row="t=%5.1f odom(%.1f,%.1f) map age %.1fs | nearest occupied %.2f m at (dx %+.2f, dy %+.2f)"%(rel,p[1],p[2],abs(mt-t),near,dx,dy)
    for R in (0.34,0.44,0.55):
        row+=" | R=%.2f: occ %d unk %d"%(R,int((occ&(D<R)).sum()),int((unk&(D<R)).sum()))
    print(row)
    # which directions are clear for 0.44 body moving 1.0 m? sample headings
    clear=[]
    for deg in range(-180,180,30):
        a=math.radians(deg); ok=True
        for s in np.arange(0.1,1.01,0.1):
            cx,cy=p[1]+s*math.cos(a),p[2]+s*math.sin(a)
            if ((occ|unk)&(np.hypot(X-cx,Y-cy)<0.44)).any(): ok=False; break
        if ok: clear.append(deg)
    print("          headings (odom frame, deg) where a 0.44 m body can move 1 m through free cells only:",clear if clear else "NONE")
