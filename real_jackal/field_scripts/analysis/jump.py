import sys, glob, math
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
R=sys.argv[1]
for B in sorted(glob.glob(R+"/bag_*")):
    if B.endswith(".log"): continue
    try:
        r=SequentialReader(); r.open(StorageOptions(uri=B,storage_id="sqlite3"),ConverterOptions("cdr","cdr"))
    except Exception as e: continue
    ty={t.name:t.type for t in r.get_all_topics_and_types()}
    if "/tf" not in ty: continue
    S=[]
    while r.has_next():
        tp,d,t=r.read_next()
        if tp!="/tf": continue
        for tr in deserialize_message(d,get_message(ty[tp])).transforms:
            if tr.header.frame_id=="map" and tr.child_frame_id=="odom":
                q=tr.transform.rotation; S.append((t*1e-9,tr.transform.translation.x,tr.transform.translation.y,math.degrees(math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y**2+q.z**2)))))
    if not S: continue
    t0=S[0][0]; jumps=[(s[0]-t0,math.hypot(s[1]-p[1],s[2]-p[2])) for p,s in zip(S,S[1:]) if math.hypot(s[1]-p[1],s[2]-p[2])>0.15]
    print("%-22s %3.0f s  map->odom start (%.2f,%.2f,%.0f deg) end (%.2f,%.2f,%.0f deg)  steps>0.15 m: %d  %s"%(B.split("/")[-1],S[-1][0]-t0,S[0][1],S[0][2],S[0][3],S[-1][1],S[-1][2],S[-1][3],len(jumps),[("t=%.0f:%.2f"%j) for j in jumps[:6]]))
