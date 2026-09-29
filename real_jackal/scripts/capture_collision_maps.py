#!/usr/bin/env python3
"""Read-only ROS subscribers; capture maps/odom and existing prediction logs.

No application publishers, service calls, parameter writes, motion commands or daemon restarts.
Output is a new gzip JSON artifact. Run in the existing project ROS shell.
"""
import argparse
from collections import deque
import gzip
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--log", default="/root/jackal_logs/spubert_real_jackal_diagnostics.jsonl")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 180:
        parser.error("--seconds must be between 1 and 180")
    if Path(args.output).exists():
        parser.error("output already exists; choose a new filename")

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, qos_profile_sensor_data
    from nav_msgs.msg import OccupancyGrid, Odometry

    rclpy.init()
    node = Node("capstone_readonly_sensitivity_capture")
    maps = deque(maxlen=240)
    odom = deque(maxlen=20000)

    def stamp_ns(header):
        return int(header.stamp.sec) * 10**9 + int(header.stamp.nanosec)

    def on_map(msg):
        maps.append({"stamp_ns": stamp_ns(msg.header), "frame_id": msg.header.frame_id,
                     "resolution": msg.info.resolution, "width": msg.info.width,
                     "height": msg.info.height,
                     "origin_xy": [msg.info.origin.position.x, msg.info.origin.position.y],
                     "origin_quaternion": [msg.info.origin.orientation.x, msg.info.origin.orientation.y,
                                           msg.info.origin.orientation.z, msg.info.origin.orientation.w],
                     "occupancy": list(msg.data)})

    def on_odom(msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        odom.append({"stamp_ns": stamp_ns(msg.header), "frame_id": msg.header.frame_id,
                     "receipt_ns": node.get_clock().now().nanoseconds,
                     "xy": [p.x, p.y], "quaternion": [q.x, q.y, q.z, q.w]})

    map_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
    subscriptions = [node.create_subscription(OccupancyGrid, "/spu_bert/debug_rolling_map", on_map, map_qos),
                     node.create_subscription(Odometry, "/aft_mapped_to_init", on_odom, qos_profile_sensor_data)]
    try:
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    records = []
    if maps:
        lo, hi = min(m["stamp_ns"] for m in maps), max(m["stamp_ns"] for m in maps)
        with open(args.log, encoding="utf-8") as source:
            for line in source:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if record.get("event") == "prediction" and lo - 5e9 <= record.get("stamp_ns", 0) <= hi + 10e9:
                    records.append(record)
    result = {"schema_version": 1, "maps": list(maps), "odom": list(odom), "predictions": records,
              "note": "Read-only capture. Map/prediction association must be validated offline; odom is not the model's exact sampled history."}
    with gzip.open(args.output, "xt", encoding="utf-8") as output:
        json.dump(result, output, allow_nan=False)
    print(f"Captured maps={len(maps)} odom={len(odom)} predictions={len(records)} -> {args.output}")


if __name__ == "__main__":
    main()
