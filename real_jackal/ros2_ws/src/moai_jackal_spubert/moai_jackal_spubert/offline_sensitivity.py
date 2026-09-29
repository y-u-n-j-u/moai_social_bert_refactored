"""Offline only: never imported by control nodes and never changes parameters."""
from __future__ import annotations

import math
from collections import Counter
import numpy as np

from .guided_spubert_runtime import heading_from_history
from .navigation_core import interpolate_polyline, normalize_angle, yaw_from_quaternion
from .rolling_laser_map import RollingLaserMapProvider


def heading_probe(points, yaw):
    heading = heading_from_history(points, yaw)  # actual unmodified production function
    displacement = math.dist(points[-2], points[-1]) if len(points) >= 2 else 0.0
    return {"displacement_m": displacement, "source": "last_two_positions" if displacement > 1e-3 else "odom_yaw",
            "heading_deg": math.degrees(heading),
            "heading_minus_yaw_deg": math.degrees(normalize_angle(heading - yaw))}


def synthetic_heading_cases():
    cases = [("below_boundary_lateral", (0.0, 0.0009)),
             ("above_boundary_lateral", (0.0, 0.0011)),
             ("above_boundary_backward", (-0.0011, 0.0)),
             ("above_boundary_opposite_lateral", (0.0, -0.0011)),
             ("forward_motion", (0.04, 0.0))]
    return [{"case": name, **heading_probe([(0.0, 0.0), point], 0.0)} for name, point in cases]


def provider_from_map(snapshot):
    resolution = float(snapshot["resolution"])
    width, height = int(snapshot["width"]), int(snapshot["height"])
    if resolution <= 0 or width < 1 or height < 1 or width != height:
        raise ValueError("expected a square rolling map with positive dimensions")
    if not np.allclose(snapshot.get("origin_quaternion", [0, 0, 0, 1]), [0, 0, 0, 1]):
        raise ValueError("rotated rolling grid is unsupported")
    data = np.asarray(snapshot["occupancy"])
    if data.size != width * height or not np.isin(data, [-1, 0, 100]).all():
        raise ValueError("expected exact debug-map cells: -1/0/100")
    provider = RollingLaserMapProvider(size_m=width * resolution, resolution=resolution,
                                       free_gap_fill_m=0, unknown_is_occupied=True)
    provider.width, provider.height = width, height
    provider.grid = np.where(data.reshape(height, width) == -1, 0,
                             np.where(data.reshape(height, width) == 0, 1, 2)).astype(np.uint8)
    provider.origin_x, provider.origin_y = map(float, snapshot["origin_xy"])
    provider.ready = True
    return provider


def match_map(record, maps):
    """No nearest-time guessing: require one map inside this prediction's build window."""
    context = record.get("collision_context")
    if not context:
        return None, "missing_collision_context"
    matches = []
    for grid in maps:
        if not context["map_update_stamp_ns"] <= grid["stamp_ns"] <= record["stamp_ns"]:
            continue
        if grid["frame_id"] != context["frame_id"]:
            continue
        if [grid["width"], grid["height"]] != context.get("map_shape_cells"):
            continue
        if not math.isclose(grid["resolution"], context["map_resolution_m"], rel_tol=1e-6):
            continue
        if not np.allclose(grid["origin_xy"], context["map_origin_xy"], rtol=0, atol=1e-8):
            continue
        matches.append(grid)
    if len(matches) != 1:
        return None, "missing_or_ambiguous_map"
    return matches[0], None


def collision_details(provider, current, path, radius):
    swept = interpolate_polyline([current, *path], max(provider.resolution * 0.5, 0.02))
    hits = [p for p in swept if provider.path_collision_cost([p], radius=radius, weight=1) > 0]
    return {"count": len(hits), "first_xy": list(hits[0]) if hits else None,
            "at_current": provider.path_collision_cost([current], radius=radius, weight=1) > 0}


def compare_record(record, snapshot, radii):
    context = record["collision_context"]
    if context.get("unknown_is_occupied") is not True:
        raise ValueError("unexpected unknown policy")
    # OccupancyGrid resolution is float32; replay with the exact float from
    # the original collision context to avoid changed cell/sample boundaries.
    provider = provider_from_map({**snapshot, "resolution": context["map_resolution_m"]})
    current = context["robot_pose"][:2]
    baseline = context["footprint_radius_m"]
    rows = []
    for attempt in record["attempts"]:
        evidence = attempt.get("collision", {})
        if not evidence.get("swept_check_performed"):
            rows.append({"rank": attempt.get("rank"), "skipped": "swept_check_not_performed"})
            continue
        path = attempt["path"]
        if not all(x is not None and y is not None and math.isfinite(x) and math.isfinite(y) for x, y in path):
            raise ValueError("nonfinite candidate")
        actual = collision_details(provider, current, path, baseline)
        if actual["count"] != evidence["footprint_collision_count"] or actual["at_current"] != evidence["current_footprint_collision"]:
            raise ValueError("baseline collision result mismatch: refuse sensitivity conclusions")
        expected_first = evidence.get("first_collision_point")
        if (actual["first_xy"] is None) != (expected_first is None) or (
            expected_first is not None and not np.allclose(actual["first_xy"], expected_first, rtol=0, atol=1e-8)
        ):
            raise ValueError("baseline first collision mismatch")
        rows.append({"rank": attempt["rank"], "original_reason": attempt["reason"],
                     "by_radius": {str(radius): collision_details(provider, current, path, radius)
                                   for radius in radii}})
    return {"time": record["stamp_ns"] / 1e9, "original_valid": record["valid"],
            "baseline_radius_m": baseline, "baseline_verified": True, "candidates": rows}


def analyze_capture(capture, radii=(0.34, 0.40, 0.45, 0.50, 0.55)):
    if not radii or any(not math.isfinite(r) or r <= 0 for r in radii):
        raise ValueError("positive finite offline radii required")
    rows, skipped = [], Counter()
    for record in capture["predictions"]:
        grid, reason = match_map(record, capture["maps"])
        if reason:
            skipped[reason] += 1
            continue
        try:
            rows.append(compare_record(record, grid, radii))
        except (ValueError, KeyError) as exc:
            skipped[str(exc)] += 1
    totals = {}
    for radius in radii:
        tested = [a for row in rows for a in row["candidates"] if "by_radius" in a]
        totals[str(radius)] = {
            "tested_candidates": len(tested),
            "collision_candidates": sum(a["by_radius"][str(radius)]["count"] > 0 for a in tested),
            "previous_collision_now_collision_free": sum(a["original_reason"] == "robot_footprint_collision"
                and a["by_radius"][str(radius)]["count"] == 0 for a in tested),
        }
    return {"warning": "Collision-only sensitivity of fixed candidates, NOT safe-to-drive or full model rerun. Other rejection checks still apply.",
            "matched_predictions": len(rows), "skipped": dict(skipped), "totals": totals, "predictions": rows}


def sampled_odom_heading_probe(odom, sample_period=0.4):
    """Proxy only: sample captured receipt order; not the bridge's exact timer history."""
    sampled = []
    for item in sorted(odom, key=lambda p: p["receipt_ns"]):
        if not sampled or item["receipt_ns"] - sampled[-1]["receipt_ns"] >= sample_period * 1e9:
            sampled.append(item)
    output = []
    for previous, current in zip(sampled, sampled[1:]):
        if previous["frame_id"] != current["frame_id"]:
            continue
        yaw = yaw_from_quaternion(*current["quaternion"])
        output.append({"time": current["receipt_ns"] / 1e9,
                       **heading_probe([previous["xy"], current["xy"]], yaw)})
    return {"warning": "0.4s resampling proxy; not exact recorded model theta and not proof of inference causality.",
            "pairs": len(output), "position_heading_pairs": sum(p["source"] == "last_two_positions" for p in output),
            "max_abs_heading_minus_yaw_deg": max((abs(p["heading_minus_yaw_deg"]) for p in output), default=None),
            "samples": output}
