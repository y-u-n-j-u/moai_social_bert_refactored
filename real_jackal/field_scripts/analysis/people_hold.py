import sys, math, collections
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
r=SequentialReader(); r.open(StorageOptions(uri=sys.argv[1],storage_id="sqlite3"),ConverterOptions("cdr","cdr"))
ty={t.name:t.type for t in r.get_all_topics_and_types()}; tr=[]; t0=None
while r.has_next():
    tp,d,t=r.read_next()
    if t0 is None: t0=t*1e-9
    if tp=="/ped_tracking": tr.append((t*1e-9-t0,deserialize_message(d,get_message(ty[tp]))))
for a,b in ((176,181),(181,187),(187,193),(193,198)):
    L=[(s,k) for s,m in tr if a<=s<b for k in m.tracks]
    if not L: print("t=%d-%d: no person tracked"%(a,b)); continue
    dm=min(math.hypot(k.pose.position.x,k.pose.position.y) for s,k in L); k=min(L,key=lambda x:math.hypot(x[1].pose.position.x,x[1].pose.position.y))[1]
    print("t=%d-%d: people tracked in %.0f%% of frames, nearest %.2f m at base (%.2f,%.2f), ids %s"%(a,b,100*len({round(s,1) for s,_ in L})/max(len([1 for s,m in tr if a<=s<b]),1),dm,k.pose.position.x,k.pose.position.y,sorted({int(k.id) for _,k in L})))
