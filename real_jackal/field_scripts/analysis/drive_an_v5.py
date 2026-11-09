import sys, math, json, collections, statistics as st
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
B=sys.argv[1]; A0=float(sys.argv[2]); A1=float(sys.argv[3])
r=SequentialReader(); r.open(StorageOptions(uri=B,storage_id="sqlite3"),ConverterOptions("cdr","cdr"))
ty={t.name:t.type for t in r.get_all_topics_and_types()}; D=collections.defaultdict(list); t0=None
keep=("/j100_0519/cmd_vel","/aft_mapped_to_init","/spu_bert/selection_evidence","/spu_bert/selection_markers","/ped_tracking")
while r.has_next():
    tp,d,t=r.read_next()
    if t0 is None: t0=t*1e-9
    if tp in keep: D[tp].append((t*1e-9-t0,deserialize_message(d,get_message(ty[tp]))))
od=[(s,m.pose.pose.position.x,m.pose.pose.position.y) for s,m in D["/aft_mapped_to_init"]]
seg=[x for x in od if A0<=x[0]<=A1]; dist=sum(math.dist(seg[i][1:],seg[i+1][1:]) for i in range(len(seg)-1))
cmd=[(s,m) for s,m in D["/j100_0519/cmd_vel"] if A0<=s<=A1]
zero=sum(1 for s in range(int(A0),int(A1)) if (lambda L:L and sum(x.linear.x for _,x in L)/len(L)<0.03)([x for x in cmd if s<=x[0]<s+1]))
rot=math.degrees(sum(abs(m.angular.z) for _,m in cmd)*(A1-A0)/max(len(cmd),1))
print("AUTONOMOUS %.0f-%.0f s (%.0f s): travelled %.1f m (mean %.2f m/s); commanded-zero seconds %d (%.0f%%); rotation %.0f deg = %.0f deg per 10 m"%(A0,A1,A1-A0,dist,dist/(A1-A0),zero,100*zero/(A1-A0),rot,rot/max(dist/10,1e-3)))
ev=[(s,json.loads(m.data)) for s,m in D["/spu_bert/selection_evidence"] if A0<=s<=A1]
src=collections.Counter(d["executed_source"] for _,d in ev)
V=sum(1 for _,d in ev for c in d["candidates"] if c["valid"]); I=sum(1 for _,d in ev for c in d["candidates"] if not c["valid"])
allinv=[d for _,d in ev if d["candidates"] and not any(c["valid"] for c in d["candidates"])]
print("cycles %d | executed_source %s"%(len(ev),dict(src)))
print("model candidates valid %d invalid %d (%.0f%% invalid) | cycles with ALL invalid: %d (%.0f%%)"%(V,I,100*I/max(V+I,1),len(allinv),100*len(allinv)/max(len(ev),1)))
print("in the ALL-INVALID cycles the published path was:",dict(collections.Counter(d["executed_source"] for d in allinv)))
fb=[d for _,d in ev if d.get("fallback_candidates")]
print("\ncycles with fallback candidates recorded: %d"%len(fb))
kinds=collections.Counter(); valid=collections.Counter(); sel=collections.Counter(); gaps=[]; inval=collections.Counter()
for d in fb:
    for r_ in d["fallback_candidates"]:
        kinds[r_["kind"]]+=1; valid[r_["valid"]]+=1
        if r_["selected"]: sel[r_["kind"]]+=1
        if not r_["valid"]: inval[r_["reason"]]+=1
        if r_["kind"]=="mgp" and r_["path"]: gaps.append(math.dist(r_["path"][-1],r_["goal"]))
print("fallback rows: %s | valid %s | SELECTED by kind %s | invalid reasons %s"%(dict(kinds),dict(valid),dict(sel),dict(inval)))
if gaps: print("MGP curve end to its goal sample: median %.3f m, max %.3f m (n=%d)"%(st.median(gaps),max(gaps),len(gaps)))
pub_fb=[d for d in fb if d["executed_source"]=="fallback"]; pub_ret=[d for d in fb if d["executed_source"]=="retained"]; pub_hold=[d for d in fb if d["executed_source"]=="hold"]
print("of those cycles: fallback published %d | previous path kept (candidates shown, none selected) %d | hold %d"%(len(pub_fb),len(pub_ret),len(pub_hold)))
nums=[0 if len([c["candidate_index"] for c in d["candidates"]]+[r_["candidate_index"] for r_ in d["fallback_candidates"]])==len(set([c["candidate_index"] for c in d["candidates"]]+[r_["candidate_index"] for r_ in d["fallback_candidates"]])) else 1 for d in fb]
print("cycles with a duplicate C number: %d"%sum(nums))
# red lines per marker message and redraw interval
sm=[(s,m) for s,m in D["/spu_bert/selection_markers"] if A0<=s<=A1]
red=[sum(1 for k in m.markers if k.ns in ("model_candidates","fallback_candidates") and k.color.r>0.9 and k.color.g<0.3) for s,m in sm]
sig=[hash(tuple(sorted((k.ns,k.id,round(k.points[-1].x,1),round(k.points[-1].y,1)) for k in m.markers if k.ns in ("model_candidates","fallback_candidates") and k.points))) for s,m in sm]
ch=1+sum(1 for i in range(1,len(sig)) if sig[i]!=sig[i-1])
print("RViz: red lines per message max %d (limit 2) | candidate set redrawn %d times in %.0f s = once per %.1f s"%(max(red),ch,A1-A0,(A1-A0)/ch))
tr=[(s,m) for s,m in D["/ped_tracking"] if A0<=s<=A1]; dm=[min(math.hypot(k.pose.position.x,k.pose.position.y) for k in m.tracks) for s,m in tr if m.tracks]
print("people: in view %.0f%% of frames, min distance %s"%(100*len(dm)/max(len(tr),1),"%.2f m"%min(dm) if dm else "none"))
