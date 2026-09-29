#!/usr/bin/env python3
"""Print compact prediction evidence from JSONL; read-only, no ROS dependency."""
import argparse
from collections import deque
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--last", type=int, default=10)
    parser.add_argument("--since", type=float, default=float("-inf"), help="ROS stamp seconds")
    parser.add_argument("--until", type=float, default=float("inf"), help="ROS stamp seconds")
    args = parser.parse_args()
    if args.last < 1:
        parser.error("--last must be positive")
    records = deque(maxlen=args.last)
    with open(args.path, encoding="utf-8") as source:
        for line in source:
            try:
                record = json.loads(line)
            except ValueError:
                continue  # includes an incomplete final line in a live log
            if not isinstance(record, dict) or record.get("event") != "prediction":
                continue
            stamp = record.get("stamp_ns")
            if not isinstance(stamp, (int, float)):
                continue
            stamp_s = stamp / 1e9
            if not args.since <= stamp_s <= args.until:
                continue
            context = record.get("collision_context", {})
            records.append({
                "time": round(stamp_s, 3), "valid": record.get("valid"),
                "reason": record.get("reason"), "selected_rank": record.get("selected_rank"),
                "collision_diagnostics_available": bool(context),
                "robot_pose": context.get("robot_pose"),
                "footprint_radius_m": context.get("footprint_radius_m"),
                "scan": context.get("scan"),
                "candidates": [{
                    "rank": a.get("rank"), "reason": a.get("reason"),
                    "checked": a.get("collision", {}).get("swept_check_performed"),
                    "at_current": a.get("collision", {}).get("current_footprint_collision"),
                    "first_xy": a.get("collision", {}).get("first_collision_point"),
                    "segment": a.get("collision", {}).get("first_collision_segment_index"),
                    "distance_m": a.get("collision", {}).get("first_collision_distance_m"),
                } for a in record.get("attempts", [])],
            })
    for record in records:
        print(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
    if not records:
        print("No prediction records in this time range.")


if __name__ == "__main__":
    main()
