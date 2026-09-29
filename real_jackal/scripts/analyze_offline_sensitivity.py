#!/usr/bin/env python3
"""Analyze saved collision captures and the production heading function offline."""
import argparse
import gzip
import json
from pathlib import Path
import sys

source = Path(__file__).resolve().parents[1] / "ros2_ws/src/moai_jackal_spubert"
if source.is_dir():
    sys.path.insert(0, str(source))
from moai_jackal_spubert.offline_sensitivity import (
    analyze_capture, sampled_odom_heading_probe, synthetic_heading_cases,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", help="gzip JSON from capture_collision_maps.py")
    parser.add_argument("--radii", nargs="+", type=float, default=[0.34, 0.40, 0.45, 0.50, 0.55])
    parser.add_argument("--output", help="new full JSON report path; never overwrites")
    args = parser.parse_args()
    report = {"synthetic_heading": synthetic_heading_cases()}
    if args.capture:
        with gzip.open(args.capture, "rt", encoding="utf-8") as source:
            capture = json.load(source)
        report["radius_comparison"] = analyze_capture(capture, args.radii)
        report["captured_odom_proxy"] = sampled_odom_heading_probe(capture.get("odom", []))
    if args.output:
        with open(args.output, "x", encoding="utf-8") as output:
            json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
    compact = {"synthetic_heading": report["synthetic_heading"]}
    if args.capture:
        compact["radius_comparison"] = {k: v for k, v in report["radius_comparison"].items() if k != "predictions"}
        compact["captured_odom_proxy"] = {k: v for k, v in report["captured_odom_proxy"].items() if k != "samples"}
    print(json.dumps(compact, ensure_ascii=False, indent=2, allow_nan=False))
    if args.output:
        print(f"Full report: {args.output}")


if __name__ == "__main__":
    main()
