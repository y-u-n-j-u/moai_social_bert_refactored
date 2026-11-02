import sys, math, collections
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
B=sys.argv[1]; r=SequentialReader(); r.open(StorageOptions(uri=B,storage_id="sqlite3"),ConverterOptions("cdr","cdr"))
ty={t.name:t.type for t in r.get_all_topics_and_types()}; ev=collections.defaultdict(list)
while r.has_next():
    tp,d,t=r.read_next()
    if tp in ("/j100_0519/cmd_vel","/spu_bert/tracker_status","/ped_tracking","/aft_mapped_to_init","/spu_bert/final_goal_odom"): ev[tp].append((t*1e-9,deserialize_message(d,get_message(ty[tp]))))
cmd=ev["/j100_0519/cmd_vel"]; ts=[t for t,m in cmd if abs(m.linear.x)>0.01][0]
print("--- status changes (not 'tracking'), merged:")
prev=None
for t,m in ev["/spu_bert/tracker_status"]:
    key=m.data.split(" ")[0] if not m.data.startswith("tracking") else "tracking"
    if key!=prev: print("  t=%5.1f  %s"%(t-ts,m.data[:90])); prev=key
print("--- nearest person every 1 s (base_link):")
last=-9
for t,m in ev["/ped_tracking"]:
    if t-last>=1.0 and t>=ts-1 and t-ts<=46:
        last=t
        if m.tracks:
            k=min(m.tracks,key=lambda k:math.hypot(k.pose.position.x,k.pose.position.y))
            x,y=k.pose.position.x,k.pose.position.y
            print("  t=%5.1f id %3d at (%5.2f,%5.2f) dist %.2f bearing %4.0f deg  v %.2f"%(t-ts,k.id,x,y,math.hypot(x,y),math.degrees(math.atan2(y,x)),math.hypot(k.twist.linear.x,k.twist.linear.y)))
        else: print("  t=%5.1f no person"%(t-ts))
print("--- cmd_vel speed v at 1 s steps:", " ".join("%d:%.2f"%(t-ts,m.linear.x) for i,(t,m) in enumerate(cmd) if i%15==0 and 0<=t-ts<=46))
g=ev["/spu_bert/final_goal_odom"][-1][1].pose.position; o=ev["/aft_mapped_to_init"][-1][1].pose.pose.position
print("last odom (%.2f,%.2f) goal (%.2f,%.2f): %.2f m"%(o.x,o.y,g.x,g.y,math.hypot(o.x-g.x,o.y-g.y)))
