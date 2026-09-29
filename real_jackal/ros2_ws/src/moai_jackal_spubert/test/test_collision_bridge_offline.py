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


def test_all_candidates_rejected_records_geometry_context_and_sends_empty_path(bridge):
    bridge._on_timer()
    assert bridge.path_pub.messages[-1].poses == []
    assert bridge._last_status == "hold:all_candidates_rejected:robot_footprint_collision"
    record = json.loads(bridge._diagnostics_file.getvalue())
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
    records = [json.loads(line) for line in bridge._diagnostics_file.getvalue().splitlines()]
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
    assert len(bridge.warnings) == 2


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
    record = json.loads(bridge._diagnostics_file.getvalue())
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
