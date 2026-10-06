"""Actual bridge callbacks with ROS/model doubles: no DDS, GPU or robot access."""
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

from test_tracker_offline import Marker as BaseMarker, Node as BaseNode, header, pose


class Marker(BaseMarker):
    DELETEALL, TEXT_VIEW_FACING = 3, 9


class Node(BaseNode):
    def declare_parameter(self, name, default):
        return super().declare_parameter(name, "" if name == "diagnostics_path" else default)


class Runtime:
    def __init__(self, **kwargs):
        self.obs_len, self.pred_len = 8, 12
        self.candidates = []

    def predict_candidates(self, **kwargs):
        self.last_kwargs = kwargs
        return self.candidates


@pytest.fixture
def bridge(monkeypatch):
    modules = {
        "rclpy": {}, "rclpy.node": {"Node": Node},
        "rclpy.executors": {"ExternalShutdownException": RuntimeError},
        "rclpy.action": {"ActionClient": lambda *a: NS()},
        "rclpy.duration": {"Duration": lambda seconds: NS(to_msg=lambda: NS(sec=int(seconds), nanosec=0))},
        "rclpy.time": {"Time": NS},
        "rclpy.qos": {"QoSDurabilityPolicy": NS(TRANSIENT_LOCAL=1),
                      "QoSReliabilityPolicy": NS(RELIABLE=1), "QoSProfile": NS,
                      "qos_profile_sensor_data": object()},
        "geometry_msgs.msg": {"Point": NS, "PoseStamped": NS},
        "moai_nav_msgs.msg": {"Tracks": NS},
        "nav2_msgs.action": {"ComputePathToPose": NS},
        "nav_msgs.msg": {"OccupancyGrid": NS, "Odometry": NS, "Path": NS},
        "sensor_msgs.msg": {"LaserScan": NS},
        "std_msgs.msg": {"ColorRGBA": NS, "String": NS},
        "tf2_ros": {"Buffer": NS, "TransformException": RuntimeError,
                    "TransformListener": lambda *a: None},
        "visualization_msgs.msg": {"Marker": Marker, "MarkerArray": lambda: NS(markers=[])},
        "moai_jackal_spubert.guided_spubert_runtime": {
            "GuidedInferenceResult": NS, "GuidedSpubertRuntime": Runtime,
            "adaptive_guidance_point_along_path": lambda *a, **kw: (2.0, 0.0),
            "lateral_guidance_point": lambda *a, **kw: None,
            "sample_polyline": lambda points, **kw: points},
    }
    for name, attrs in modules.items():
        parent = name.split(".")[0]
        if parent != "moai_jackal_spubert" and parent not in modules:
            monkeypatch.setitem(sys.modules, parent, ModuleType(parent))
        mod = ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)
    file = Path(__file__).parents[1] / "moai_jackal_spubert/real_jackal_spubert_bridge_node.py"
    spec = importlib.util.spec_from_file_location("moai_jackal_spubert._offline_bridge", file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    node = mod.RealJackalSpubertBridgeNode()
    node._diagnostics_file = io.StringIO()
    node._on_odom(NS(header=header(time_s=99.9), pose=NS(pose=pose())))
    node._on_scan(NS(header=header("base_link", 99.7), ranges=[3.0],
                     angle_min=0.0, angle_increment=0.1, range_min=0.1, range_max=10.0))
    node._on_tracks(NS(header=header("base_link"), tracks=[]))
    node._goal = NS()
    node._robot_history.extend([(0, 0)] * 8)
    node.route_refresh_period = 0
    node._pose_in_frame = lambda *a: (5.0, 0.0)
    node._route_in_frame = lambda *a: [(0, 0), (5, 0)]
    node._publish_debug_map = lambda *a: None
    node._publish_markers = lambda *a: None
    node._path_message = lambda frame, points: NS(frame=frame, poses=list(points))
    node._point_pose = lambda frame, point: NS(frame=frame, point=point)
    node.map_collision = True

    def update(**kwargs):
        node.used_map_inputs = kwargs
        node.map_provider.ready = True
        node.map_provider.grid.fill(2 if node.map_collision else 1)

    node.map_provider.update_from_scan = update
    node._runtime.candidates = [NS(
        selected_goal_valid=True, trajectory_map_safe=True, execution_valid=True,
        path_world=[(0.2 * (i + 1), 0.01 * rank) for i in range(12)],
        candidate_rank=rank, candidate_index=rank + 10,
    ) for rank in range(5)]
    return node


def _predictions(bridge):
    return [json.loads(line) for line in bridge._diagnostics_file.getvalue().splitlines()
            if json.loads(line)["event"] == "prediction"]


def test_all_candidates_rejected_records_geometry_context_and_sends_empty_path(bridge):
    bridge._on_timer()
    assert bridge.path_pub.messages[-1].poses == []
    assert bridge._last_status == "hold:all_candidates_rejected:robot_footprint_collision"
    record = _predictions(bridge)[0]
    assert record["valid"] is False
    assert len(record["attempts"]) == 5
    assert all(a["collision"]["current_footprint_collision"] for a in record["attempts"])
    assert all(len(a["path"]) == 12 for a in record["attempts"])
    context = record["collision_context"]
    assert context["robot_pose"] == [0.0, 0.0, 0.0]
    assert context["footprint_radius_m"] == 0.5
    assert context["scan"]["source_age_s"] == pytest.approx(0.3)
    assert context["scan"]["received_age_s"] == 0
    assert context["odom"]["source_age_s"] == pytest.approx(0.1)
    assert bridge.used_map_inputs["robot_yaw"] == context["robot_pose"][2]
    markers = bridge.collision_marker_pub.messages[-1].markers
    assert markers[0].action == Marker.DELETEALL
    assert len(markers) == 11
    center = markers[1]
    assert center.color.b == 1.0  # magenta: current robot pose rejected
    assert center.header.frame_id == "odom"
    assert center.lifetime.sec == 2


def test_recovery_selects_same_first_valid_candidate_and_clears_markers(bridge):
    bridge._on_timer()
    bridge.map_collision = False
    bridge.now_s += 1.0
    bridge._odom_stamp_s = bridge._scan_stamp_s = bridge._tracks_stamp_s = bridge.now_s
    bridge._on_timer()
    records = _predictions(bridge)
    assert [r["valid"] for r in records] == [False, True]
    assert records[-1]["selected_rank"] == 0
    assert bridge.path_pub.messages[-1].poses == bridge._runtime.candidates[0].path_world
    assert len(bridge.collision_marker_pub.messages[-1].markers) == 1


def test_diagnostics_io_and_marker_failures_cannot_prevent_empty_path_stop(bridge):
    def fail(*args):
        raise OSError("simulated diagnostic failure")
    bridge.collision_marker_pub.publish = fail
    bridge._diagnostics_file = NS(write=fail)
    bridge._on_timer()
    assert bridge.path_pub.messages[-1].poses == []
    assert len(bridge.warnings) >= 2  # marker + every diagnostics write that failed


def test_non_start_marker_is_red_and_sensor_hold_clears_it(bridge):
    bridge._publish_collision_markers("odom", [{"rank": 2, "collision": {
        "first_collision_point": [1, 2], "first_collision_segment_index": 3,
        "current_footprint_collision": False}}])
    markers = bridge.collision_marker_pub.messages[-1].markers
    assert markers[1].color.r == 1 and markers[1].color.b == 0
    assert markers[2].text == "rank=2 seg=3"
    bridge._hold("scan_missing_or_stale")
    assert bridge.path_pub.messages[-1].poses == []
    assert len(bridge.collision_marker_pub.messages[-1].markers) == 1


def test_unchecked_model_rejection_is_explicit_and_nonfinite_path_logs_safely(bridge):
    for result in bridge._runtime.candidates:
        result.selected_goal_valid = False
        result.path_world[0] = (float("nan"), 0)
    bridge._on_timer()
    record = _predictions(bridge)[0]
    assert record["reason"] == "no_map_safe_goal"
    assert record["attempts"][0]["collision"] == {"swept_check_performed": False}
    assert record["attempts"][0]["path"][0] == [None, 0]
    assert bridge.path_pub.messages[-1].poses == []


def test_lateral_detour_used_when_along_path_search_fails(bridge):
    bridge_globals = type(bridge)._on_timer.__globals__

    def raise_no_safe_point(*args, **kwargs):
        raise ValueError("no directly footprint-safe guidance point on global path")

    bridge_globals["adaptive_guidance_point_along_path"] = raise_no_safe_point
    bridge_globals["lateral_guidance_point"] = lambda *a, **kw: (1.0, 0.6)
    bridge.map_collision = False  # candidates are otherwise safe once guided sideways

    bridge._on_timer()

    assert bridge._last_status.startswith("path_valid")
    assert bridge.path_pub.messages[-1].poses == bridge._runtime.candidates[0].path_world
    assert bridge._runtime.last_kwargs["guidance_point_world"] == (1.0, 0.6)
    assert bridge._blocked_since_s is None
    record = json.loads(bridge._diagnostics_file.getvalue().splitlines()[0])
    assert record["event"] == "lateral_detour"
    assert record["reason"] == "adaptive_guidance_failed_used_lateral_detour"
    assert record["robot"] == [0.0, 0.0, 0.0]
    assert record["guidance"] == [1.0, 0.6]


def test_short_block_still_holds_empty_like_before(bridge):
    bridge._on_timer()
    assert bridge._last_status == "hold:all_candidates_rejected:robot_footprint_collision"
    assert bridge.path_pub.messages[-1].poses == []

    bridge.now_s += 1.0  # under stuck_hold_timeout_sec (3.0s): still a plain hold
    bridge._odom_stamp_s = bridge._scan_stamp_s = bridge._tracks_stamp_s = bridge.now_s
    bridge._on_timer()
    assert bridge._last_status == "hold:all_candidates_rejected:robot_footprint_collision"
    assert bridge.path_pub.messages[-1].poses == []


def test_blocked_hold_escalates_to_recovery_sweep_after_timeout(bridge):
    bridge._request_route = lambda: False  # isolate from the action-client plumbing
    bridge.recovery_rotation_clearance_m = 0.0  # fully occupied test map: guard covered below
    bridge._on_timer()

    bridge.now_s += bridge.stuck_hold_timeout_sec + 0.5
    bridge._odom_stamp_s = bridge._scan_stamp_s = bridge._tracks_stamp_s = bridge.now_s
    bridge._on_timer()

    assert bridge._last_status.startswith("recovery_sweep:all_candidates_rejected:")
    path = bridge.path_pub.messages[-1]
    assert len(path.poses) == 2
    assert path.poses[0] == (0.0, 0.0)
    tx, ty = path.poses[1]
    assert (tx ** 2 + ty ** 2) ** 0.5 == pytest.approx(bridge.recovery_step_m, abs=1e-6)

    # A fresh valid candidate ends recovery and goes straight back to normal tracking.
    bridge.map_collision = False
    bridge.now_s += 1.0  # clear replan_period's throttle too
    bridge._odom_stamp_s = bridge._scan_stamp_s = bridge._tracks_stamp_s = bridge.now_s
    bridge._on_timer()
    assert bridge.path_pub.messages[-1].poses == bridge._runtime.candidates[0].path_world
    assert bridge._blocked_since_s is None
    assert bridge._recovery_active is False


# --- global-path fallback -------------------------------------------------

def _model_rejects_everything(bridge):
    for result in bridge._runtime.candidates:
        result.selected_goal_valid = False  # model-side rejection, map is free
    bridge.map_collision = False


def _events(bridge, name):
    records = [json.loads(line) for line in bridge._diagnostics_file.getvalue().splitlines()]
    return [r for r in records if r["event"] == name]


def test_safe_route_prefix_is_followed_when_model_rejects_all_candidates(bridge):
    _model_rejects_everything(bridge)
    bridge._on_timer()
    assert bridge._last_status.startswith("path_valid global_fallback")
    path = bridge.path_pub.messages[-1].poses
    assert len(path) == 12
    assert path[0][0] > 0 and all(abs(y) < 1e-9 for _, y in path)  # along the route
    assert path[-1][0] == pytest.approx(1.5)
    assert bridge._blocked_since_s is None and not bridge._recovery_active
    event = _events(bridge, "global_path_fallback")[0]
    assert event["trigger"] == "all_candidates_rejected:no_map_safe_goal"
    assert event["points_kept"] == 12


def test_unsafe_route_is_never_followed_and_reason_is_logged(bridge):
    for result in bridge._runtime.candidates:
        result.selected_goal_valid = False
    bridge.map_collision = True  # live obstacle covers the whole route
    bridge._on_timer()
    assert bridge._last_status == "hold:all_candidates_rejected:no_map_safe_goal"
    assert bridge.path_pub.messages[-1].poses == []
    assert not _events(bridge, "global_path_fallback")
    rejected = _events(bridge, "global_path_fallback_rejected")
    assert rejected and rejected[0]["reason"] == "robot_footprint_collision"


def test_only_the_safe_part_of_the_route_is_used_but_not_below_the_minimum(bridge):
    _model_rejects_everything(bridge)
    bridge.arc_planner_fallback = False  # this test is about the route prefix alone
    real_check = type(bridge)._try_global_path_fallback.__globals__["validate_candidate_path"]

    def blocked_beyond(limit):
        def check(**kw):
            if kw["path"][-1][0] > limit:
                return NS(valid=False, reason="robot_footprint_collision",
                          minimum_human_distance_m=float("inf"))
            return real_check(**kw)
        return check

    globals_ = type(bridge)._try_global_path_fallback.__globals__
    try:
        globals_["validate_candidate_path"] = blocked_beyond(1.0)
        bridge._on_timer()
        path = bridge.path_pub.messages[-1].poses
        assert bridge._last_status.startswith("path_valid global_fallback")
        assert 0.8 <= path[-1][0] <= 1.0

        bridge.now_s += 1.0
        bridge._odom_stamp_s = bridge._scan_stamp_s = bridge._tracks_stamp_s = bridge.now_s
        globals_["validate_candidate_path"] = blocked_beyond(0.5)  # < 0.8 m safe
        bridge._on_timer()
        assert bridge.path_pub.messages[-1].poses == []
        assert bridge._last_status.startswith("hold:")
    finally:
        globals_["validate_candidate_path"] = real_check


def test_nearby_pedestrian_blocks_the_fallback_like_any_model_path(bridge):
    _model_rejects_everything(bridge)
    bridge._human_histories = {7: [(1.0, 0.0)] * 8}
    bridge._on_timer()
    assert bridge.path_pub.messages[-1].poses == []
    assert bridge._last_status.startswith("hold:")
    assert not _events(bridge, "global_path_fallback")


def test_fallback_stops_after_the_episode_time_limit(bridge):
    _model_rejects_everything(bridge)
    bridge._request_route = lambda: False  # isolate recovery from the action client
    bridge._on_timer()
    assert bridge._last_status.startswith("path_valid global_fallback")
    for _ in range(3):  # keep the episode alive with fresh sensor data
        bridge.now_s += bridge.global_fallback_max_sec / 2
        bridge._odom_stamp_s = bridge._scan_stamp_s = bridge._tracks_stamp_s = bridge.now_s
        bridge._global_fallback_last_s = bridge.now_s - 0.5
        bridge._on_timer()
    assert not bridge._last_status.startswith("path_valid")  # normal recovery took over
    assert len(bridge.path_pub.messages[-1].poses) != 12  # not a route-prefix path
    assert _events(bridge, "global_path_fallback_rejected")[-1]["reason"] == "episode_time_limit"


def test_fallback_can_be_disabled(bridge):
    _model_rejects_everything(bridge)
    bridge.global_path_fallback = False
    bridge._on_timer()
    assert bridge.path_pub.messages[-1].poses == []
    assert bridge._last_status.startswith("hold:")


def test_model_path_ends_the_fallback_episode(bridge):
    _model_rejects_everything(bridge)
    bridge._on_timer()
    assert bridge._global_fallback_since_s is not None
    for result in bridge._runtime.candidates:
        result.selected_goal_valid = True
    bridge.now_s += 1.0
    bridge._odom_stamp_s = bridge._scan_stamp_s = bridge._tracks_stamp_s = bridge.now_s
    bridge._on_timer()
    assert bridge._last_status.startswith("path_valid rank=")
    assert bridge._global_fallback_since_s is None


def test_failed_lateral_detour_is_logged_then_global_fallback_is_tried(bridge):
    globals_ = type(bridge)._on_timer.__globals__

    def raise_no_safe_point(*args, **kwargs):
        raise ValueError("no directly footprint-safe guidance point on global path")

    globals_["adaptive_guidance_point_along_path"] = raise_no_safe_point
    globals_["lateral_guidance_point"] = lambda *a, **kw: None
    bridge.map_collision = False
    bridge._on_timer()
    assert _events(bridge, "lateral_detour_failed")
    assert bridge._last_status.startswith("path_valid global_fallback")
    assert _events(bridge, "global_path_fallback")[0]["trigger"].startswith("adaptive_guidance_failed")


# --- narrow passage: box on the left, wall on the right, ~0.1 m safe tube ---

def _narrow_passage(bridge, wall_y=-0.68):
    """Live obstacle 0.43 m left of the centre line and a wall at ``wall_y``."""
    def cost(points, radius, weight=1.0):
        for x, y in points:
            if 0.3 <= x <= 1.4 and (0.43 - y < radius or y - wall_y < radius):
                return 1.0
        return 0.0
    bridge.map_provider.path_collision_cost = cost
    bridge.map_collision = False
    _model_rejects_everything(bridge)
    return cost


def test_route_prefix_is_slid_into_a_narrow_passage(bridge):
    cost = _narrow_passage(bridge)
    assert cost([(0.6, 0.0)], radius=0.5) > 0  # the straight route is blocked
    bridge._on_timer()
    assert bridge._last_status.startswith("path_valid global_fallback")
    event = _events(bridge, "global_path_fallback")[0]
    assert -0.18 < event["lateral_shift_m"] < -0.05  # right, a few cm only
    path = bridge.path_pub.messages[-1].poses
    assert len(path) >= 7
    assert cost([(0.0, 0.0), *path], radius=0.5) == 0.0  # the very check that gates publishing
    assert path[0][0] > 0 and abs(path[0][1]) < 0.1  # attached to the robot


def test_no_passage_means_no_fallback_even_with_sideways_search(bridge):
    _narrow_passage(bridge, wall_y=-0.40)  # only 0.83 m gap: no centre line fits 0.5 m radius
    bridge.arc_planner_fallback = False  # this test is about the sideways route search alone
    bridge.escape_footprint_radius_m = 0.0
    bridge._on_timer()
    assert bridge.path_pub.messages[-1].poses == []
    assert bridge._last_status.startswith("hold:")
    assert not _events(bridge, "global_path_fallback")
    assert _events(bridge, "global_path_fallback_rejected")


# --- model-free arc planner: obstacle sitting on the route ------------------

def _obstacle_on_route(bridge, obstacle_radius=0.1, centre=(1.0, 0.0), wall=False):
    def cost(points, radius, weight=1.0):
        for x, y in points:
            if wall and (x + radius > 0.55 or abs(y) + radius > 0.60):
                return 1.0  # a pocket the robot fits in but cannot leave
            if not wall and (x - centre[0]) ** 2 + (y - centre[1]) ** 2 < (obstacle_radius + radius) ** 2:
                return 1.0
        return 0.0
    bridge.map_provider.path_collision_cost = cost
    bridge.map_collision = False
    _model_rejects_everything(bridge)
    return cost


def test_arc_goes_around_an_obstacle_that_sideways_shift_cannot_clear(bridge):
    cost = _obstacle_on_route(bridge)  # blocks radius 0.6 around (1, 0); shift max is only 0.3
    bridge._on_timer()
    assert bridge._last_status.startswith("path_valid global_fallback source=arc")
    event = _events(bridge, "global_path_fallback")[0]
    assert event["source"] == "arc" and event["curvature"] != 0.0
    path = bridge.path_pub.messages[-1].poses
    assert cost([(0.0, 0.0), *path], radius=0.5) == 0.0  # same check that gates publishing
    assert max(abs(y) for _, y in path) > 0.6  # really goes around, not through
    assert path[-1][0] > 1.0  # and makes progress toward the goal at (5, 0)


def test_pedestrian_on_one_side_makes_the_arc_choose_the_other(bridge):
    _obstacle_on_route(bridge)
    bridge._human_histories = {3: [(1.0, 0.9)] * 8}
    bridge._on_timer()
    assert bridge._last_status.startswith("path_valid global_fallback source=arc")
    path = bridge.path_pub.messages[-1].poses
    assert path[-1][1] < -0.3 and all(y < 0.3 for _, y in path)  # passes on the far side


def test_no_arc_when_the_way_is_completely_blocked(bridge):
    _obstacle_on_route(bridge, wall=True)
    bridge._request_route = lambda: False
    bridge._on_timer()
    assert not bridge._last_status.startswith("path_valid")
    assert not _events(bridge, "global_path_fallback")
    assert _events(bridge, "global_path_fallback_rejected")


def test_arc_planner_can_be_disabled(bridge):
    _obstacle_on_route(bridge)
    bridge.arc_planner_fallback = False
    bridge._request_route = lambda: False
    bridge._on_timer()
    assert not bridge._last_status.startswith("path_valid")
    assert not _events(bridge, "global_path_fallback")


def test_arc_still_works_when_the_robot_is_too_far_from_the_route_to_use_it(bridge):
    _obstacle_on_route(bridge, obstacle_radius=0.05, centre=(1.0, 3.0))  # obstacle irrelevant to the arc
    bridge._route_in_frame = lambda *a: [(0, 4.0), (5, 4.0)]  # route 4 m away: no prefix
    bridge._on_timer()
    assert bridge._last_status.startswith("path_valid global_fallback source=arc")


# --- recovery rotation must not swing the body into an obstacle -------------

def _sweep_scene(bridge, nearest_obstacle_m):
    """Map whose only obstacle is `nearest_obstacle_m` from the robot centre."""
    def cost(points, radius, weight=1.0):
        return 1.0 if radius > nearest_obstacle_m else 0.0
    bridge.map_provider.path_collision_cost = cost
    bridge.map_collision = False
    bridge.escape_footprint_radius_m = 0.0  # these tests are about the rotation, not the escape tier
    for result in bridge._runtime.candidates:
        result.selected_goal_valid = False
    bridge._request_route = lambda: False
    bridge._on_timer()
    bridge.now_s += bridge.stuck_hold_timeout_sec + 0.5
    bridge._odom_stamp_s = bridge._scan_stamp_s = bridge._tracks_stamp_s = bridge.now_s


def test_recovery_sweep_runs_when_the_rotation_circle_is_clear(bridge):
    _sweep_scene(bridge, nearest_obstacle_m=0.45)  # body circle (0.38) is clear
    bridge._on_timer()
    assert bridge._last_status.startswith("recovery_sweep:")
    assert len(bridge.path_pub.messages[-1].poses) == 2


def test_recovery_sweep_is_blocked_when_an_obstacle_is_inside_the_rotation_circle(bridge):
    _sweep_scene(bridge, nearest_obstacle_m=0.30)  # a corner would swing into it
    bridge._on_timer()
    assert bridge._last_status.startswith("hold:recovery_blocked_obstacle_in_rotation_circle:")
    assert bridge.path_pub.messages[-1].poses == []
    blocked = _events(bridge, "recovery_sweep_blocked")[0]
    assert blocked["rotation_clearance_m"] == 0.38
    assert blocked["clearance_probe"]["0.30"] is False and blocked["clearance_probe"]["0.34"] is True
    assert not _events(bridge, "recovery_sweep")


def test_rejected_fallback_reports_how_close_the_robot_already_is(bridge):
    _sweep_scene(bridge, nearest_obstacle_m=0.30)
    rejected = _events(bridge, "global_path_fallback_rejected")[0]
    probe = rejected["clearance_probe"]
    assert probe["0.30"] is False and all(probe[k] for k in ("0.34", "0.38", "0.44", "0.50"))


# --- smooth avoidance: no pivots by default, escape tier instead of deadlock ---

def test_default_arc_start_directions_never_force_a_stop_and_pivot(bridge):
    assert bridge.arc_heading_offsets
    assert max(abs(v) for v in bridge.arc_heading_offsets) < 0.95  # tracker rotate_in_place_angle
    assert 0.0 in bridge.arc_heading_offsets


def _parallel_obstacle(bridge, nearest_m):
    """Obstacle `nearest_m` from the centre line along the whole drive: nothing fits at a larger radius."""
    def cost(points, radius, weight=1.0):
        return 1.0 if radius > nearest_m else 0.0
    bridge.map_provider.path_collision_cost = cost
    bridge.map_collision = False
    _model_rejects_everything(bridge)
    bridge._request_route = lambda: False


def test_escape_tier_keeps_the_robot_moving_when_the_normal_margin_deadlocks(bridge):
    _parallel_obstacle(bridge, nearest_m=0.40)  # normal footprint is 0.50, escape 0.36
    bridge._on_timer()
    assert bridge._last_status.startswith("path_valid global_fallback")
    assert "tier=escape" in bridge._last_status
    assert _events(bridge, "global_path_fallback")[0]["tier"] == "escape"
    assert len(bridge.path_pub.messages[-1].poses) >= 7


def test_escape_tier_is_not_used_when_the_robot_is_closer_than_the_escape_radius(bridge):
    _parallel_obstacle(bridge, nearest_m=0.30)  # already inside 0.36: stay put
    bridge._on_timer()
    assert not bridge._last_status.startswith("path_valid")
    assert not _events(bridge, "global_path_fallback")
    assert "escape:" not in _events(bridge, "global_path_fallback_rejected")[0]["reason"]  # never attempted


def test_normal_margin_is_preferred_over_the_escape_tier(bridge):
    _parallel_obstacle(bridge, nearest_m=0.60)  # everything fits at the normal radius
    bridge._on_timer()
    assert bridge._last_status.startswith("path_valid global_fallback")
    assert "tier=normal" in bridge._last_status


def test_narrow_passage_that_only_fits_the_reduced_radius_is_driven_in_the_escape_tier(bridge):
    _narrow_passage(bridge, wall_y=-0.40)  # 0.83 m gap: fits a 0.36 m radius, not 0.50 m
    bridge.arc_planner_fallback = False
    bridge._on_timer()
    assert bridge._last_status.startswith("path_valid global_fallback")
    assert "tier=escape" in bridge._last_status
    assert abs(_events(bridge, "global_path_fallback")[0]["lateral_shift_m"]) <= 0.07  # centre line -0.04..+0.07 m is free


def test_escape_radius_adapts_to_how_close_the_robot_already_is(bridge):
    _parallel_obstacle(bridge, nearest_m=0.345)  # between the 0.33 floor and the 0.36 cap
    bridge._on_timer()
    assert bridge._last_status.startswith("path_valid global_fallback")
    event = _events(bridge, "global_path_fallback")[0]
    assert event["tier"] == "escape"
    assert 0.33 <= event["footprint_radius_m"] <= 0.345  # follows the robot's clearance, not a fixed 0.36


def test_escape_never_goes_below_the_body_radius_floor(bridge):
    _parallel_obstacle(bridge, nearest_m=0.32)  # closer than the 0.33 body circle: do not drive
    bridge._on_timer()
    assert not bridge._last_status.startswith("path_valid")
    assert not _events(bridge, "global_path_fallback")


def test_escape_paths_do_not_get_closer_to_the_obstacle_than_the_robot_already_is(bridge):
    # obstacle 0.35 m to the side now, but only 0.30 m away further ahead: moving on would
    # be closer than the start, so the adaptive radius (>= 0.35) must reject that path.
    def cost(points, radius, weight=1.0):
        for x, y in points:
            nearest = 0.35 if x < 0.2 else 0.30
            if radius > nearest:
                return 1.0
        return 0.0
    bridge.map_provider.path_collision_cost = cost
    bridge.map_collision = False
    _model_rejects_everything(bridge)
    bridge._request_route = lambda: False
    bridge._on_timer()
    assert not bridge._last_status.startswith("path_valid global_fallback")


# --- continuity: the published path must not flip sideways between cycles ---

def _swerving_and_straight(bridge):
    straight = [(0.2 * (i + 1), 0.0) for i in range(12)]
    swerve = [(0.02 * (i + 1), 0.1 * (i + 1)) for i in range(12)]  # ~80 deg to the left
    bridge._runtime.candidates = [
        NS(selected_goal_valid=True, trajectory_map_safe=True, execution_valid=True,
           path_world=swerve, candidate_rank=0, candidate_index=10),
        NS(selected_goal_valid=True, trajectory_map_safe=True, execution_valid=True,
           path_world=straight, candidate_rank=1, candidate_index=11),
    ]
    bridge.map_collision = False
    return straight, swerve


def test_default_selection_is_still_the_first_valid_candidate(bridge):
    straight, swerve = _swerving_and_straight(bridge)
    bridge._on_timer()
    assert bridge.path_pub.messages[-1].poses == swerve  # old behaviour, weights are 0 by default


def test_sharp_swerve_is_skipped_when_a_straighter_valid_candidate_exists(bridge):
    straight, swerve = _swerving_and_straight(bridge)
    bridge.selection_config = type(bridge.selection_config)(
        rank_weight=0.15, continuity_weight=2.0, heading_weight=0.3, heading_limit_rad=0.9)
    bridge._on_timer()
    assert bridge.path_pub.messages[-1].poses == straight
    assert _predictions(bridge)[0]["selected_rank"] == 1  # logged as the rank actually used
