"""Exercise real tracker callbacks with in-memory ROS doubles (no DDS or robot).

This is not a substitute for ROS message/type support or real-robot testing.
The fake Node never opens a socket and timers execute only when called here.
"""
import copy
import importlib.util
import json
import math
from pathlib import Path as FilePath
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest


def vector():
    return NS(x=0.0, y=0.0, z=0.0)


def header(frame="odom", time_s=100.0):
    return NS(frame_id=frame, stamp=NS(sec=int(time_s), nanosec=int((time_s % 1) * 1e9)))


def pose(x=0.0, y=0.0):
    return NS(position=NS(x=x, y=y, z=0.0), orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0))


class Twist:
    def __init__(self):
        self.linear = vector()
        self.angular = vector()


class Marker:
    ADD, DELETE, SPHERE = 0, 2, 2

    def __init__(self):
        self.header = header()
        self.pose = pose()
        self.scale = vector()
        self.color = NS(r=0.0, g=0.0, b=0.0, a=0.0)
        self.lifetime = NS(sec=0, nanosec=0)


class Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(copy.deepcopy(message))


class Node:
    def __init__(self, name):
        self.now_s = 100.0
        self.parameters = {}
        self.warnings = []

    def declare_parameter(self, name, default):
        self.parameters[name] = default
        return NS(value=default)

    def create_publisher(self, *args):
        return Publisher()

    def create_subscription(self, *args):
        pass

    def create_service(self, *args):
        pass

    def create_timer(self, *args):
        pass

    def get_clock(self):
        return NS(now=lambda: NS(nanoseconds=int(self.now_s * 1e9),
                                 to_msg=lambda: header(time_s=self.now_s).stamp))

    def get_logger(self):
        return NS(info=lambda *a, **kw: None,
                  warning=lambda text, **kw: self.warnings.append(text))


def make_path(points, frame="odom", time_s=100.0):
    return NS(header=header(frame, time_s), poses=[NS(pose=pose(x, y)) for x, y in points])


@pytest.fixture
def tracker(monkeypatch):
    modules = {
        "rclpy": {}, "rclpy.executors": {"ExternalShutdownException": RuntimeError},
        "rclpy.node": {"Node": Node}, "rclpy.qos": {"qos_profile_sensor_data": object()},
        "geometry_msgs.msg": {"PoseStamped": NS, "Twist": Twist},
        "moai_nav_msgs.msg": {"Tracks": NS},
        "nav_msgs.msg": {"Odometry": NS, "Path": NS},
        "sensor_msgs.msg": {"LaserScan": NS},
        "std_msgs.msg": {"Bool": NS, "String": NS},
        "std_srvs.srv": {"SetBool": NS}, "visualization_msgs.msg": {"Marker": Marker},
    }
    for name, attributes in modules.items():
        for parent in [name.split(".")[0], name]:
            if parent not in modules or parent == name:
                module = ModuleType(parent)
                if parent == name:
                    module.__dict__.update(attributes)
                monkeypatch.setitem(sys.modules, parent, module)
    file = FilePath(__file__).parents[1] / "moai_jackal_spubert" / "safe_path_tracker_node.py"
    spec = importlib.util.spec_from_file_location("moai_jackal_spubert._offline_tracker", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    node = module.SafePathTrackerNode()
    node._on_odom(NS(header=header(), pose=NS(pose=pose(0.39, 0.05)), twist=NS(twist=Twist())))
    node._on_scan(NS(header=header("base_link"), ranges=[10.0], angle_min=0.0, angle_increment=0.1))
    node._on_tracks(NS(header=header("base_link"), tracks=[]))
    node._on_goal(NS(header=header(), pose=pose(4.0, 0.0)))
    node._on_path(make_path([(0, 0), (0.8, 0), (1.6, 0), (2.4, 0)]))
    return node


def record(tracker):
    return json.loads(tracker.diagnostics_pub.messages[-1].data)


def assert_zero(tracker):
    command = tracker.cmd_pub.messages[-1]
    assert command.linear.x == command.angular.z == 0.0


def test_startup_stays_disarmed_and_safety_defaults_unchanged(tracker):
    tracker._control_tick()
    assert_zero(tracker)
    assert record(tracker)["status"] == "disarmed"
    assert not record(tracker)["motion_enabled"]
    assert tracker.cmd_vel_topic == "/spu_bert/cmd_vel_dryrun"
    assert tracker.obstacle_stop_distance == 0.75
    assert tracker.human_stop_distance == 0.90
    assert tracker.path_timeout == 1.20


def test_real_control_uses_forward_projection_and_reports_limited_command(tracker):
    tracker.motion_enabled = True  # Only this in-memory test instance is armed.
    tracker._control_tick()
    data = record(tracker)
    assert data["lookahead"]["target"] == pytest.approx([1.09, 0.0])
    assert data["lookahead"]["projection"] == pytest.approx([0.39, 0.0])
    assert data["target_command"]["v"] > 0.24
    assert data["published_command"]["v"] == pytest.approx(0.035)
    assert data["published_command"]["w"] == tracker.cmd_pub.messages[-1].angular.z
    assert tracker.lookahead_pub.messages[-1].action == Marker.ADD
    assert tracker.lookahead_pub.messages[-1].lifetime.nanosec == 500_000_000


def test_angular_diagnostics_distinguish_target_from_rate_limited_output(tracker):
    tracker.motion_enabled = True
    tracker._odom.pose.pose.position.y = 0.5
    tracker._control_tick()
    assert record(tracker)["target_command"]["w"] == -0.55
    assert record(tracker)["published_command"]["w"] == pytest.approx(-0.12)


def test_empty_path_zeroes_motion_and_preserves_first_reason(tracker):
    tracker.motion_enabled = True
    tracker._control_tick()
    tracker._on_path(make_path([]))
    assert_zero(tracker)
    tracker.now_s += 0.1
    tracker._control_tick()
    data = record(tracker)
    assert data["status"] == "predicted_path_missing_or_stale"
    assert data["stop_episode"]["first_reason"] == "planner_hold"
    assert data["lookahead"] is None
    assert not data["path"]["present"]
    assert tracker.lookahead_pub.messages[-1].action == Marker.DELETE
    tracker._on_path(make_path([(0, 0), (3, 0)]))
    tracker._control_tick()
    assert record(tracker)["stop_episode"]["first_reason"] is None
    assert record(tracker)["stop_episode"]["last_completed"]["first_reason"] == "planner_hold"


@pytest.mark.parametrize("field, reason", [
    ("_odom_stamp_s", "odom_missing_or_stale"),
    ("_scan_stamp_s", "scan_missing_or_stale"),
    ("_tracks_stamp_s", "pedestrian_tracker_heartbeat_stale"),
    ("_path_stamp_s", "predicted_path_missing_or_stale"),
])
def test_freshness_guards_still_stop(tracker, field, reason):
    tracker.motion_enabled = True
    setattr(tracker, field, tracker.now_s - 2.0)
    tracker._control_tick()
    assert_zero(tracker)
    assert record(tracker)["status"] == reason


@pytest.mark.parametrize("case, reason", [
    ("human", "human_too_close"), ("obstacle", "obstacle_too_close"),
    ("scan_frame", "scan_frame_mismatch"), ("tracks_frame", "tracks_frame_mismatch"),
    ("path_frame", "frame_mismatch"), ("goal", "goal_reached_disarmed"),
    ("pose_nan", "invalid_robot_pose"), ("degenerate", "path_has_no_lookahead"),
    ("endpoint", "path_endpoint_reached"),
    ("past_endpoint", "path_endpoint_reached"),
])
def test_other_stop_guards_and_degenerate_target(tracker, case, reason):
    tracker.motion_enabled = True
    if case == "human":
        tracker._minimum_human_distance = 0.5
    elif case == "obstacle":
        tracker._scan.ranges = [0.5]
    elif case == "scan_frame":
        tracker._scan_frame_valid = False
    elif case == "tracks_frame":
        tracker._tracks_frame_valid = False
    elif case == "path_frame":
        tracker._on_path(make_path([(0, 0), (3, 0)], frame="map"))
    elif case == "goal":
        tracker._goal.pose.position.x = 0.39
    elif case == "pose_nan":
        tracker._odom.pose.pose.position.x = math.nan
    elif case == "degenerate":
        tracker._on_path(make_path([(0, 0), (0, 0)]))
    elif case == "endpoint":
        tracker._on_path(make_path([(0, 0.05), (0.39, 0.05)]))
    elif case == "past_endpoint":
        tracker._on_path(make_path([(0, 0.05), (0.3, 0.05)]))
    tracker._control_tick()
    assert_zero(tracker)
    assert record(tracker)["status"].startswith(reason)
    if case == "goal":
        assert not tracker.motion_enabled


def test_emergency_latch_cannot_be_overridden_by_new_path(tracker):
    tracker.motion_enabled = True
    tracker._on_emergency_stop(NS(data=True))
    tracker._on_path(make_path([(0, 0), (3, 0)]))
    response = tracker._on_enable_motion(NS(data=True), NS())
    assert not response.success
    tracker._control_tick()
    assert_zero(tracker)
    assert record(tracker)["emergency_latched"]


def test_new_geometry_resets_progress_but_identical_republish_does_not(tracker):
    tracker.motion_enabled = True
    tracker._control_tick()
    geometry_id = record(tracker)["path"]["geometry_id"]
    tracker._on_path(make_path([(0, 0), (0.8, 0), (1.6, 0), (2.4, 0)], time_s=99.8))
    assert tracker._progress_tracker.progress == pytest.approx(0.39)
    tracker._control_tick()
    data = record(tracker)
    assert data["path"]["geometry_id"] == geometry_id
    assert data["path"]["receipt_id"] == 2
    assert data["path"]["source_age_s"] == pytest.approx(0.2)
    assert data["path"]["received_age_s"] == 0
    tracker._on_path(make_path([(0, 0), (4, 0)]))
    assert tracker._progress_tracker.progress is None


def test_nonfinite_path_is_rejected_immediately(tracker):
    tracker.motion_enabled = True
    tracker._on_path(make_path([(0, 0), (math.inf, 0)]))
    assert_zero(tracker)
    assert record(tracker)["status"] == "invalid_path_coordinates"


def test_diagnostic_publisher_failure_does_not_prevent_stop(tracker):
    tracker.motion_enabled = True
    tracker._control_tick()

    def fail(message):
        raise RuntimeError("observer unavailable")

    tracker.diagnostics_pub.publish = fail
    tracker._on_path(make_path([]))
    assert_zero(tracker)
    assert "diagnostics unavailable" in tracker.warnings[-1]


# --- corridor obstacle rule (narrow-passage fix) ----------------------------

def scan_with_point(tracker, x, y):
    """Scan of empty space (inf) with a single return at (x, y) in base_link."""
    step = 0.01
    count = int(2 * math.pi / step) + 1
    ranges = [math.inf] * count
    index = round((math.atan2(y, x) + math.pi) / step)
    ranges[index] = math.hypot(x, y)
    tracker._on_scan(NS(header=header("base_link"), ranges=ranges,
                        angle_min=-math.pi, angle_increment=step))


def use_corridor_rule(tracker):
    tracker.obstacle_corridor_half_width = 0.42
    tracker.obstacle_stop_distance = 0.45
    tracker.obstacle_slow_distance = 1.00
    tracker.obstacle_hard_stop_distance = 0.38


def test_sector_rule_stops_for_a_box_edge_beside_the_robot(tracker):
    tracker.motion_enabled = True
    scan_with_point(tracker, 0.60, 0.44)  # 0.74 m away, 32 deg off to the side
    tracker._control_tick()
    assert_zero(tracker)
    assert record(tracker)["status"].startswith("obstacle_too_close")


def test_corridor_rule_lets_the_robot_pass_a_box_edge_beside_it(tracker):
    use_corridor_rule(tracker)
    tracker.motion_enabled = True
    scan_with_point(tracker, 0.60, 0.44)  # outside the +-0.42 m body corridor
    tracker._control_tick()
    assert record(tracker)["status"].startswith("tracking")
    assert tracker.cmd_pub.messages[-1].linear.x > 0.0


def test_corridor_rule_still_stops_for_an_obstacle_in_the_path(tracker):
    use_corridor_rule(tracker)
    tracker.motion_enabled = True
    scan_with_point(tracker, 0.40, 0.10)  # dead ahead, inside 0.45 m
    tracker._control_tick()
    assert_zero(tracker)
    assert record(tracker)["status"].startswith("obstacle_too_close:0.40")


def test_corridor_rule_keeps_the_emergency_stop_very_close_in_the_sector(tracker):
    use_corridor_rule(tracker)
    tracker.motion_enabled = True
    scan_with_point(tracker, 0.30, 0.10)  # r = 0.32 m < hard stop 0.38 m
    tracker._control_tick()
    assert_zero(tracker)
    assert record(tracker)["status"].startswith("obstacle_too_close")


def test_corridor_rule_slows_for_obstacles_ahead_between_stop_and_slow_distance(tracker):
    use_corridor_rule(tracker)
    tracker.motion_enabled = True
    scan_with_point(tracker, 0.80, 0.0)
    tracker._control_tick()
    slowed = record(tracker)["target_command"]["v"]
    scan_with_point(tracker, 5.0, 0.0)
    tracker._control_tick()
    clear = record(tracker)["target_command"]["v"]
    assert 0.10 < slowed < 0.20 < 0.24 < clear
