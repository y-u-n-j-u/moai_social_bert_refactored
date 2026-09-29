import copy
import math

import pytest

from moai_jackal_spubert.offline_sensitivity import (
    analyze_capture, collision_details, heading_probe, match_map, provider_from_map,
    sampled_odom_heading_probe, synthetic_heading_cases,
)
from moai_jackal_spubert.rolling_laser_map import RollingLaserMapProvider


def example_capture():
    provider = RollingLaserMapProvider(size_m=8, resolution=0.1)
    provider.grid.fill(1)
    # Horizontal wall at y=1m; the path inclines towards it without touching it.
    provider.grid[50, :] = 2
    path = [[0.4, 0.45], [1.0, 0.45], [2.0, 0.45]]
    baseline = collision_details(provider, [0, 0], path, 0.5)
    grid = {"stamp_ns": 110, "frame_id": "odom", "resolution": 0.10000000149011612,
            "width": 80, "height": 80, "origin_xy": [-4, -4],
            "occupancy": provider.occupancy_grid_data()}
    context = {"map_update_stamp_ns": 100, "frame_id": "odom", "robot_pose": [0, 0, 0],
               "map_shape_cells": [80, 80], "map_resolution_m": 0.1,
               "map_origin_xy": [-4, -4], "unknown_is_occupied": True, "footprint_radius_m": 0.5}
    attempt = {"rank": 1, "reason": "robot_footprint_collision" if baseline["count"] else "valid",
               "path": path, "collision": {"swept_check_performed": True,
                   "footprint_collision_count": baseline["count"],
                   "current_footprint_collision": baseline["at_current"],
                   "first_collision_point": baseline["first_xy"]}}
    return {"maps": [grid], "predictions": [{"stamp_ns": 200, "valid": baseline["count"] == 0,
            "collision_context": context, "attempts": [attempt]}]}


def test_one_tenth_mm_boundary_crossing_can_rotate_heading_90_degrees():
    results = synthetic_heading_cases()
    assert results[0]["heading_deg"] == 0
    assert results[1]["heading_deg"] == 90
    assert results[2]["heading_deg"] == 180
    assert results[3]["heading_deg"] == -90
    assert results[4]["heading_deg"] == 0
    assert results[0]["source"] == "odom_yaw"
    assert results[1]["source"] == "last_two_positions"


def test_exact_threshold_falls_back_to_yaw_and_errors_wrap():
    assert heading_probe([(0, 0), (0, .001)], 0.3)["heading_deg"] == pytest.approx(math.degrees(.3))
    assert heading_probe([], -.2)["heading_deg"] == pytest.approx(math.degrees(-.2))
    result = heading_probe([(0, 0), (-0.01, 0)], math.radians(-179))
    assert result["heading_minus_yaw_deg"] == pytest.approx(-1)


def test_same_grid_paths_replayed_at_multiple_radii_without_mutation():
    capture = example_capture()
    before = copy.deepcopy(capture)
    report = analyze_capture(capture)
    assert capture == before
    assert report["matched_predictions"] == 1
    assert report["skipped"] == {}
    totals = report["totals"]
    assert totals["0.34"]["collision_candidates"] == 0
    assert totals["0.55"]["collision_candidates"] == 1
    assert report["predictions"][0]["baseline_verified"] is True


def test_missing_or_ambiguous_map_is_never_guessed():
    capture = example_capture()
    record = capture["predictions"][0]
    assert match_map(record, [])[1] == "missing_or_ambiguous_map"
    assert match_map(record, capture["maps"] * 2)[1] == "missing_or_ambiguous_map"
    for key, value in [("frame_id", "map"), ("stamp_ns", 201), ("origin_xy", [0, 0])]:
        changed = dict(capture["maps"][0], **{key: value})
        assert match_map(record, [changed])[0] is None


def test_baseline_mismatch_prevents_sensitivity_conclusions():
    capture = example_capture()
    capture["predictions"][0]["attempts"][0]["collision"]["footprint_collision_count"] = 999
    report = analyze_capture(capture)
    assert report["matched_predictions"] == 0
    assert any("baseline collision result mismatch" in k for k in report["skipped"])


def test_unknown_cells_remain_blocked_and_invalid_grid_is_rejected():
    grid = example_capture()["maps"][0]
    grid["occupancy"] = [-1] * 6400
    provider = provider_from_map(grid)
    assert provider.point_occupied_with_radius(0, 0, 0.34)
    grid["occupancy"][0] = 50
    with pytest.raises(ValueError):
        provider_from_map(grid)


def test_unchecked_attempt_is_not_counted_as_collision_free():
    capture = example_capture()
    capture["predictions"][0]["attempts"][0]["collision"] = {"swept_check_performed": False}
    report = analyze_capture(capture)
    assert report["totals"]["0.5"]["tested_candidates"] == 0


def test_captured_odom_heading_is_explicitly_a_proxy_and_frame_changes_are_skipped():
    samples = [{"receipt_ns": int(t * 1e9), "frame_id": frame, "xy": [0, y],
                "quaternion": [0, 0, 0, 1]} for t, y, frame in
               [(0, 0, "odom"), (.1, .1, "odom"), (.4, .0011, "odom"), (.8, .1, "new")]]
    report = sampled_odom_heading_probe(samples)
    assert report["pairs"] == 1
    assert report["position_heading_pairs"] == 1
    assert report["max_abs_heading_minus_yaw_deg"] == 90
    assert "not exact" in report["warning"]


@pytest.mark.parametrize("radii", [[], [0], [-.5], [float("nan")]])
def test_invalid_offline_radii_are_rejected(radii):
    with pytest.raises(ValueError):
        analyze_capture(example_capture(), radii)
