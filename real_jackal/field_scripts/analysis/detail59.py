import sys, math, json, collections
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
r=SequentialReader(); r.open(StorageOptions(uri=sys.argv[1],storage_id="sqlite3"),ConverterOptions("cdr","cdr"))
ty={t.name:t.type for t in r.get_all_topics_and_types()}; D=collections.defaultdict(list); t0=None
keep=("/j100_0519/cmd_vel","/aft_mapped_to_init","/spu_bert/tracker_status","/spu_bert/selection_evidence","/ped_tracking")
while r.has_next():
    tp,d,t=r.read_next()
    if t0 is None: t0=t*1e-9
    if tp in keep: D[tp].append((t*1e-9-t0,deserialize_message(d,get_message(ty[tp]))))
od=[(s,m.pose.pose.position.x,m.pose.pose.position.y) for s,m in D["/aft_mapped_to_init"]]
P=lambda s:min(od,key=lambda x:abs(x[0]-s))
print("robot position / speed by second around the one hold and around x=2..3 m:")
for s in list(range(56,64)):
    a=[x for x in od if s<=x[0]<s+1]; sp=math.dist(a[0][1:],a[-1][1:]) if len(a)>1 else 0
    c=[m.linear.x for t,m in D["/j100_0519/cmd_vel"] if s<=t<s+1]; v=sum(c)/len(c) if c else 0
    tr=[(m.data[:44]) for t,m in D["/spu_bert/tracker_status"] if s<=t<s+1][-1:]
    pe=[(k.id,round(math.hypot(k.pose.position.x,k.pose.position.y),2),round(math.degrees(math.atan2(k.pose.position.y,k.pose.position.x)))) for t,m in D["/ped_tracking"] if s<=t<s+1 for k in m.tracks]
    near=min(pe,key=lambda x:x[1]) if pe else None
    print("  t=%d odom(%.1f,%.1f) odom speed %.2f cmd v %.2f status %s | nearest person (id,dist,bearing) %s"%(s,P(s)[1],P(s)[2],sp,v,tr,near))
ev=[(s,json.loads(m.data)) for s,m in D["/spu_bert/selection_evidence"] if 58<=s<=61.5]
for s,d in ev[::2]: print("  evidence t=%.1f source=%s decision=%s | cands %s"%(s,d["executed_source"],str(d.get("decision_reason"))[:50],[(c["candidate_index"],c["valid"],c["reason"]) for c in d["candidates"]][:5]))
for lo,hi in ((1.5,3.5),):
    seg=[x for x in od if lo<=x[1]<=hi]; 
    if seg: print("\nthe earlier trouble spot x=%.1f..%.1f m: passed in %.1f s (t=%.0f to %.0f)"%(lo,hi,seg[-1][0]-seg[0][0],seg[0][0],seg[-1][0]))
