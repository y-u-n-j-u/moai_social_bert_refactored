import sys, math, json, collections, numpy as np
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
B=sys.argv[1]; A0=float(sys.argv[2]); A1=float(sys.argv[3]); H0=float(sys.argv[4]); H1=float(sys.argv[5])
r=SequentialReader(); r.open(StorageOptions(uri=B,storage_id="sqlite3"),ConverterOptions("cdr","cdr"))
ty={t.name:t.type for t in r.get_all_topics_and_types()}; D=collections.defaultdict(list)
keep=("/j100_0519/cmd_vel","/aft_mapped_to_init","/spu_bert/selection_evidence","/spu_bert/debug_rolling_map","/spu_bert/final_goal_odom","/spu_bert/tracker_status")
while r.has_next():
    tp,d,t=r.read_next()
    if tp in keep: D[tp].append((t*1e-9,deserialize_message(d,get_message(ty[tp]))))
od=[(t,m.pose.pose.position.x,m.pose.pose.position.y) for t,m in D["/aft_mapped_to_init"]]; t0=od[0][0]
P=lambda s:min(od,key=lambda x:abs(x[0]-(t0+s)))
ev=[(t-t0,json.loads(m.data)) for t,m in D["/spu_bert/selection_evidence"]]
cmd=[(t-t0,m) for t,m in D["/j100_0519/cmd_vel"]]
w=[(s,d) for s,d in ev if A0<=s<=A1]
dist=sum(math.dist(od[i][1:],od[i+1][1:]) for i in range(len(od)-1) if t0+A0<=od[i][0]<=t0+A1)
c=[(s,m) for s,m in cmd if A0<=s<=A1]
print("AUTONOMOUS window %.0f-%.0f s (%.0f s): travelled %.1f m (mean %.2f m/s)"%(A0,A1,A1-A0,dist,dist/(A1-A0)))
holdsec=sum(1 for s in range(int(A0),int(A1)) if (lambda L:L and sum(x.linear.x for _,x in L)/len(L)<0.03)([x for x in c if s<=x[0]<s+1]))
print("seconds with commanded v~0 inside the window: %d (%.0f%%)"%(holdsec,100*holdsec/(A1-A0)))
rot=math.degrees(sum(abs(m.angular.z) for _,m in c)*(A1-A0)/max(len(c),1)); print("rotation %.0f deg = %.0f deg per 10 m"%(rot,rot/max(dist/10,1e-3)))
src=collections.Counter(d["executed_source"] for _,d in w); print("executed_source:",dict(src))
V=sum(1 for _,d in w for x in d["candidates"] if x["valid"]); I=sum(1 for _,d in w for x in d["candidates"] if not x["valid"])
allinv=sum(1 for _,d in w if d["candidates"] and not any(x["valid"] for x in d["candidates"])); nc=sum(1 for _,d in w if not d["candidates"])
print("cycles %d | no candidates at all: %d | candidates valid %d invalid %d | all-invalid cycles %d"%(len(w),nc,V,I,allinv))
print("invalid reasons:",dict(collections.Counter(x["reason"] for _,d in w for x in d["candidates"] if not x["valid"]).most_common(5)))
print("decision_reason overall:",dict(collections.Counter(str(d.get("decision_reason"))[:48] for _,d in w).most_common(6)))
print("\n== the long hold %.0f-%.0f s =="%(H0,H1))
h=[(s,d) for s,d in ev if H0<=s<=H1]
print("cycles %d | executed_source %s"%(len(h),dict(collections.Counter(d["executed_source"] for _,d in h))))
print("decision_reason:",dict(collections.Counter(str(d.get("decision_reason"))[:60] for _,d in h).most_common(6)))
print("candidates per cycle:",dict(sorted(collections.Counter(len(d["candidates"]) for _,d in h).items())))
print("candidate rejection reasons:",dict(collections.Counter(x["reason"] for _,d in h for x in d["candidates"] if not x["valid"]).most_common(6)))
pp=P(H0+3); print("robot odom at hold (%.2f,%.2f)"%(pp[1],pp[2]))
for s,d in h[::max(len(h)//6,1)][:6]:
    gp=d.get("guidance_point"); fg=d.get("final_goal"); rp=d["robot_pose"]
    print("  t=%5.1f robot (%.2f,%.2f) final_goal (%.2f,%.2f) dist_to_goal %.2f | guidance_point %s src %s | cands %s"%(s,rp[0],rp[1],fg[0],fg[1],math.dist(rp[:2],fg[:2]),[round(v,2) for v in gp] if gp else None,d.get("guidance_source"),[(x["candidate_index"],x["reason"],round(x["goal_progress_m"],2) if x["goal_progress_m"] is not None else None) for x in d["candidates"]][:4]))
fg=D["/spu_bert/final_goal_odom"]; print("\nfinal goal over time:",sorted({(round(m.pose.position.x,2),round(m.pose.position.y,2)) for _,m in fg}))
e=[(s,d) for s,d in ev if 202<=s<=206]; 
print("goal-reached check: robot odom at the goal_reached stop (%.2f,%.2f); final goal then %s"%(P(204)[1],P(204)[2],[round(v,2) for v in e[0][1]["final_goal"]] if e else None))
