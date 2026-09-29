"""Collision diagnostics must preserve the pre-change safety decision/samples."""
import json
import math
from types import SimpleNamespace as NS

import pytest

from moai_jackal_spubert.collision_diagnostics import collision_evidence, header_evidence
from moai_jackal_spubert.navigation_core import interpolate_polyline, validate_candidate_path
from moai_jackal_spubert.rolling_laser_map import RollingLaserMapProvider
from moai_jackal_spubert.tracker_diagnostics import diagnostic_json


class ProbeMap:
    resolution = 0.1

    def __init__(self, occupied=lambda x, y: False):
        self.occupied = occupied
        self.calls = []

    def path_collision_cost(self, path, radius, weight):
        self.calls.append((path[0], radius, weight))
        return sum(self.occupied(x, y) for x, y in path)


def validate(provider, path=((0.4, 0.0), (0.8, 0.0)), **overrides):
    args = dict(current=(0.0, 0.0), path=path, final_goal=(4.0, 0.0),
                map_provider=provider, footprint_radius=0.5, prediction_dt=0.4,
                maximum_model_speed=2.0, maximum_step_ratio=1.5, minimum_goal_progress=0.1,
                human_histories={}, human_sample_dt=0.4, minimum_human_center_distance=1.2,
                human_radius=0.35, human_safety_margin=0.2)
    args.update(overrides)
    return validate_candidate_path(**args)


def test_start_collision_shared_by_all_five_candidates():
    for y in (-0.3, -0.1, 0.0, 0.1, 0.3):
        check = validate(ProbeMap(lambda x, y: x == 0 and y == 0), ((0.4, y), (0.8, y)))
        assert not check.valid
        assert check.reason == "robot_footprint_collision"
        assert check.current_footprint_collision is True
        assert check.first_collision_point == (0.0, 0.0)
        assert check.first_collision_sample_index == 0
        assert check.first_collision_segment_index is None  # current pose, not a segment
        assert check.first_collision_distance_m == 0


def test_collision_between_waypoints_records_center_segment_and_distance():
    check = validate(ProbeMap(lambda x, y: 0.54 < x < 0.56))
    assert check.reason == "robot_footprint_collision"
    assert check.current_footprint_collision is False
    assert check.first_collision_point == pytest.approx((0.55, 0))
    assert check.first_collision_segment_index == 1  # path[0] -> path[1]
    assert check.first_collision_sample_index == 11
    assert check.first_collision_distance_m == pytest.approx(0.55)
    assert check.footprint_collision_count == 1


def test_current_to_first_waypoint_is_segment_zero():
    check = validate(ProbeMap(lambda x, y: 0.09 < x < 0.11))
    assert check.first_collision_segment_index == 0


def test_duplicates_and_corner_keep_arc_length_and_original_segment_index():
    check = validate(ProbeMap(lambda x, y: y > 0.19), ((0, 0), (0.4, 0), (0.4, 0.4)))
    assert check.first_collision_segment_index == 2
    assert check.first_collision_distance_m == pytest.approx(0.6)
    assert check.first_collision_point == pytest.approx((0.4, 0.2))


@pytest.mark.parametrize("path", [((0.4, 0), (0.8, 0)), ((0, 0), (0.4, 0.4), (0.4, 0.4))])
@pytest.mark.parametrize("boundary", [-1.0, 0.0, 0.25, 0.6, 2.0])
def test_exact_same_samples_radius_query_order_and_collision_count(path, boundary):
    provider = ProbeMap(lambda x, y: x >= boundary)
    swept = interpolate_polyline([(0, 0), *path], max(provider.resolution * 0.5, 0.02))
    # Pre-change implementation queried every swept point once, without early exit.
    expected = sum(provider.path_collision_cost([p], radius=0.5, weight=1.0) > 0 for p in swept)
    previous_calls = list(provider.calls)
    provider.calls.clear()
    check = validate(provider, path)
    assert provider.calls == previous_calls
    assert check.footprint_collision_count == expected
    assert check.swept_sample_count == len(swept)
    assert check.valid == (expected == 0)
    if expected == 0:
        assert check.first_collision_point is None
        assert check.current_footprint_collision is False


@pytest.mark.parametrize("cell_value", [0, 1, 2])
def test_real_rolling_map_unknown_free_and_occupied_keep_decisions(cell_value):
    provider = RollingLaserMapProvider(size_m=8, resolution=0.1, unknown_is_occupied=True)
    provider.grid.fill(cell_value)
    # Constructor origin is immaterial: put the robot/path in the center of this grid.
    x, y = provider.origin_x + 4, provider.origin_y + 4
    path = [(x + 0.4, y), (x + 0.8, y)]
    swept = interpolate_polyline([(x, y), *path], 0.05)
    expected = sum(provider.path_collision_cost([p], radius=0.5, weight=1.0) > 0 for p in swept)
    check = validate(provider, path, current=(x, y), final_goal=(x + 3, y))
    assert check.footprint_collision_count == expected
    assert check.valid == (cell_value == 1)


@pytest.mark.parametrize("path", [[], [(math.nan, 0)]])
def test_skipped_check_is_not_reported_as_free(path):
    provider = ProbeMap()
    check = validate(provider, path)
    assert not check.valid
    assert provider.calls == []
    assert collision_evidence(check)["swept_check_performed"] is False
    assert check.current_footprint_collision is None
    assert collision_evidence(None) == {"swept_check_performed": False}


@pytest.mark.parametrize("kwargs,reason", [
    ({"maximum_model_speed": 0.1}, "kinematic_jump"),
    ({"minimum_goal_progress": 2.0}, "insufficient_goal_progress"),
    ({"human_histories": {1: [(0.4, 0.0)]}}, "predicted_human_clearance"),
])
def test_other_rejection_reasons_and_collision_precedence_unchanged(kwargs, reason):
    assert validate(ProbeMap(), **kwargs).reason == reason
    assert validate(ProbeMap(lambda x, y: True), **kwargs).reason == "robot_footprint_collision"


def test_time_evidence_preserves_source_vs_receipt_and_negative_age():
    msg = NS(header=NS(frame_id="base_link", stamp=NS(sec=100, nanosec=250000000)))
    result = header_evidence(msg, received_s=100.4, now_s=100.5)
    assert result["source_age_s"] == 0.25
    assert result["received_age_s"] == pytest.approx(0.1)
    assert header_evidence(msg, 100, 99)["source_age_s"] == -1.25


def test_collision_evidence_and_invalid_geometry_are_strict_json():
    result = json.loads(diagnostic_json({"collision": collision_evidence(validate(ProbeMap())),
                                         "path": [[math.nan, math.inf]]}))
    assert result["collision"]["swept_check_performed"] is True
    assert result["path"] == [[None, None]]


def test_summary_cli_bounds_output_supports_old_logs_and_ignores_partial_line(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    script = Path(__file__).resolve().parents[4] / "scripts/summarize_collision_diagnostics.py"
    path = tmp_path / "records.jsonl"
    lines = [json.dumps({"event": "prediction", "stamp_ns": t * 10**9,
                        "valid": True, "attempts": []}) for t in (1, 2, 3)]
    path.write_text("\n".join([*lines, '{"event":']) + "\n")
    output = subprocess.check_output([sys.executable, str(script), str(path), "--last", "1",
                                      "--since", "1", "--until", "2"], text=True)
    result = json.loads(output)
    assert result["time"] == 2
    assert result["collision_diagnostics_available"] is False
