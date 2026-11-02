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
print("--- tracker_status around the stop and at the end")
for t,m in ev["/spu_bert/tracker_status"]:
    if 10<=t-ts<=18 or t-ts>cmd[-1][0]-ts-4: print("  t=%5.1f  %s"%(t-ts,m.data[:70]))
print("--- person (base_link) t=10..18 s, every 0.5 s")
last=-9
for t,m in ev["/ped_tracking"]:
    if 10<=t-ts<=18 and t-last>=0.5 and m.tracks:
        k=min(m.tracks,key=lambda k:math.hypot(k.pose.position.x,k.pose.position.y)); last=t
        print("  t=%5.1f id %d at (%.2f,%.2f) dist %.2f speed %.2f"%(t-ts,k.id,k.pose.position.x,k.pose.position.y,math.hypot(k.pose.position.x,k.pose.position.y),math.hypot(k.twist.linear.x,k.twist.linear.y)))
g=ev["/spu_bert/final_goal_odom"][-1][1].pose.position; o=ev["/aft_mapped_to_init"][-1][1].pose.pose.position
print("final odom (%.2f,%.2f) goal (%.2f,%.2f): %.2f m from the goal"%(o.x,o.y,g.x,g.y,math.hypot(o.x-g.x,o.y-g.y)))
