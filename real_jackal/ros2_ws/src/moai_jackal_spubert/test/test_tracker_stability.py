"""Tracker regressions that run without ROS, hardware, or model weights."""
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

from moai_jackal_spubert.navigation_core import (
    finite_ranges_in_sector,
    goal_approach_speed_limit,
    lookahead_point,
    nearest_range_in_sector,
    validate_candidate_path,
)


class LookaheadTests(unittest.TestCase):
    def assertPoint(self, actual, expected):
        self.assertIsNotNone(actual)
        for value, target in zip(actual, expected):
            self.assertAlmostEqual(value, target)

    def test_two_centimetre_motion_does_not_reverse_target_at_vertex_boundary(self):
        points = [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)]
        before = lookahead_point(points, (0.99, 0.0), 0.7)
        after = lookahead_point(points, (1.01, 0.0), 0.7)
        self.assertPoint(before, (1.69, 0.0))
        self.assertPoint(after, (1.71, 0.0))
        self.assertAlmostEqual(after[0] - before[0], 0.02)

    def test_off_path_projection_does_not_spend_lookahead_reaching_path(self):
        self.assertPoint(lookahead_point([(0, 0), (4, 0)], (1, 2), 0.7), (1.7, 0))

    def test_lookahead_follows_arc_length_around_corner(self):
        self.assertPoint(
            lookahead_point([(0, 0), (2, 0), (2, 2)], (1.8, -0.1), 0.7), (2, 0.5)
        )

    def test_exact_corner_and_duplicate_vertices_keep_forward_direction(self):
        self.assertPoint(
            lookahead_point([(0, 0), (2, 0), (2, 0), (2, 2), (2, 2)], (2, 0), 0.7),
            (2, 0.7),
        )

    def test_short_remaining_path_clamps_to_endpoint(self):
        self.assertPoint(lookahead_point([(0, 0), (2, 0)], (1.8, 0), 0.7), (2, 0))
        self.assertPoint(lookahead_point([(0, 0), (2, 0)], (2.2, 0), 0.7), (2, 0))

    def test_before_start_projects_to_start_then_walks_forward(self):
        self.assertPoint(lookahead_point([(0, 0), (2, 0)], (-1, 0.5), 0.7), (0.7, 0))

    def test_empty_single_and_all_duplicate_paths(self):
        self.assertIsNone(lookahead_point([], (0, 0), 0.7))
        for points in [[(2, 3)], [(2, 3), (2, 3), (2, 3)]]:
            self.assertPoint(lookahead_point(points, (0, 0), 0.7), (2, 3))

    def test_zero_or_negative_lookahead_returns_projection(self):
        for distance in [0, -0.5]:
            self.assertPoint(lookahead_point([(0, 0), (2, 0)], (0.8, 1), distance), (0.8, 0))

    def test_nonfinite_geometry_fails_closed(self):
        self.assertIsNone(lookahead_point([(0, 0), (math.nan, 1)], (0, 0), 0.7))
        self.assertIsNone(lookahead_point([(0, 0), (2, 0)], (math.inf, 0), 0.7))
        self.assertIsNone(lookahead_point([(0, 0), (2, 0)], (0, 0), math.nan))

    def test_execution_does_not_chase_an_exhausted_endpoint_backwards(self):
        for current in [(0.2, 0), (0.21, 0), (0.21, 0.01)]:
            self.assertIsNone(lookahead_point([(0, 0), (0.2, 0)], current, 0.7,
                                             require_forward_progress=True))
        self.assertPoint(lookahead_point([(0, 0), (0.2, 0)], (0.21, 0), 0.7), (0.2, 0))

    def test_execution_keeps_forward_corner_and_small_remaining_segments(self):
        cases = [
            ([(0, 0), (0.2, 0)], (0.19, 0), (0.2, 0)),
            ([(0, 0), (0.001, 0)], (0, 0), (0.001, 0)),
            ([(0, 0), (1, 0), (1, 0), (1, 1)], (1, 0), (1, 0.7)),
        ]
        for points, current, expected in cases:
            self.assertPoint(lookahead_point(points, current, 0.7, require_forward_progress=True), expected)
        self.assertIsNone(lookahead_point([(0.2, 0), (0.2, 0)], (0, 0), 0.7,
                                         require_forward_progress=True))


class HumanSweepTests(unittest.TestCase):
    def check(self, histories, **overrides):
        class FreeMap:
            resolution = 0.1

            def path_collision_cost(self, path, radius, weight):
                return 0

        arguments = dict(
            current=(0, 0), path=[(0.4, 0), (0.8, 0)], final_goal=(4, 0),
            map_provider=FreeMap(), footprint_radius=0.5, prediction_dt=0.4,
            maximum_model_speed=1.5, maximum_step_ratio=1.5, minimum_goal_progress=0.1,
            human_histories=histories, human_sample_dt=0.4,
            minimum_human_center_distance=1.2, human_radius=0.35, human_safety_margin=0.2,
        )
        arguments.update(overrides)
        return validate_candidate_path(**arguments)

    def test_stationary_person_can_be_too_close_between_safe_waypoints(self):
        # Both ends of the first segment are 1.20669 m away (>1.2 m), but the
        # midpoint passes at 1.19 m. The former waypoint-only check missed it.
        self.assertGreater(math.hypot(0.2, 1.19), 1.2)
        check = self.check({1: [(0.2, 1.19), (0.2, 1.19)]})
        self.assertFalse(check.valid)
        self.assertEqual(check.reason, "predicted_human_clearance")
        self.assertAlmostEqual(check.minimum_human_distance_m, 1.19)

    def test_person_outside_required_clearance_is_still_valid(self):
        check = self.check({1: [(0.2, 1.21)]})
        self.assertTrue(check.valid)
        self.assertAlmostEqual(check.minimum_human_distance_m, 1.21)

    def test_moving_person_crossing_between_samples_is_checked(self):
        check = self.check(
            {1: [(0.2, 1.5), (0.2, 0.5)]}, footprint_radius=0.1,
            minimum_human_center_distance=0.3, human_radius=0.1, human_safety_margin=0.1,
        )
        self.assertEqual(check.reason, "predicted_human_clearance")
        self.assertAlmostEqual(check.minimum_human_distance_m, 0)

    def test_equal_robot_human_velocity_and_initial_separation_are_checked(self):
        check = self.check({1: [(-0.4, 1.3), (0, 1.3)]})
        self.assertTrue(check.valid)
        self.assertAlmostEqual(check.minimum_human_distance_m, 1.3)
        check = self.check({1: [(-0.2, 1.19)]})
        self.assertAlmostEqual(check.minimum_human_distance_m, math.hypot(0.2, 1.19))
        self.assertTrue(check.valid)


class FrontSectorTests(unittest.TestCase):
    def test_diagnostic_bearing_does_not_change_existing_distance_filter(self):
        for ranges in [[math.nan, 1.0, 0.8], [math.inf, math.inf, math.inf],
                       [0.2, 0.3, 0.4], [-0.1, 2.0, math.nan]]:
            minimum, bearing = nearest_range_in_sector(ranges, -0.7, 0.7, 0.7)
            self.assertEqual(minimum, min(finite_ranges_in_sector(ranges, -0.7, 0.7, 0.7), default=math.inf))
            self.assertEqual(bearing is None, math.isinf(minimum))


class GoalApproachTests(unittest.TestCase):
    @staticmethod
    def speed(distance):
        return goal_approach_speed_limit(distance, 0.5, 1.5, 0.25, 0.05)

    def test_approach_cap_is_bounded_monotonic_and_stops_at_tolerance(self):
        distances = [0.49, 0.5, 0.500001, 0.75, 1.0, 1.5, 4.0]
        speeds = [self.speed(distance) for distance in distances]
        self.assertEqual(speeds[:2], [0, 0])
        self.assertEqual(speeds, sorted(speeds))
        self.assertGreaterEqual(speeds[2], 0.05)
        self.assertEqual(speeds[-2:], [0.25, 0.25])

    def test_nonzero_approach_speed_reaches_tolerance_in_finite_time(self):
        distance = 1.5
        for _ in range(300):
            if distance <= 0.5:
                break
            distance -= self.speed(distance) * 0.1
        self.assertLessEqual(distance, 0.5)

    def test_minimum_never_overrides_maximum_speed(self):
        self.assertEqual(goal_approach_speed_limit(0.6, 0.5, 1.5, 0.02, 0.05), 0.02)
        self.assertEqual(goal_approach_speed_limit(0.6, 0.5, 1.5, 0, 0.05), 0)


class Twist:
    def __init__(self):
        self.linear = NS(x=0.0, y=0.0, z=0.0)
        self.angular = NS(x=0.0, y=0.0, z=0.0)


class RosPath:
    def __init__(self):
        self.header = NS(frame_id="", stamp=NS(sec=0, nanosec=0))
        self.poses = []


class RosPoseStamped:
    def __init__(self):
        self.header = NS(frame_id="", stamp=NS(sec=0, nanosec=0))
        self.pose = NS(position=NS(x=0.0, y=0.0, z=0.0), orientation=NS(x=0.0, y=0.0, z=0.0, w=0.0))


class Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def ros_test_modules(parameters=None):
    """Return reusable message/transport stubs; callers may add ROS interfaces."""
    class FakeNode:
        def __init__(self, _name):
            self._test_now = 100.0
            self._test_qos = {}
            self._test_log = []

        def declare_parameter(self, name, default):
            return NS(value=(parameters or {}).get(name, default))

        def create_publisher(self, _type, topic, qos):
            self._test_qos[topic] = qos
            return Publisher()

        def create_subscription(self, _type, topic, _callback, qos):
            self._test_qos[topic] = qos

        def create_service(self, *_args):
            pass

        def create_timer(self, *_args):
            pass

        def get_clock(self):
            return NS(now=lambda: NS(nanoseconds=int(self._test_now * 1e9)))

        def get_logger(self):
            return NS(info=self._test_log.append, warning=self._test_log.append)

        def destroy_node(self):
            pass

    modules = {}
    def module(name, **attributes):
        result = ModuleType(name)
        result.__dict__.update(attributes)
        modules[name] = result

    module("rclpy")
    module("rclpy.node", Node=FakeNode)
    module("rclpy.executors", ExternalShutdownException=RuntimeError)
    module("rclpy.qos", QoSProfile=lambda **kw: NS(**kw),
           ReliabilityPolicy=NS(RELIABLE="reliable"),
           DurabilityPolicy=NS(TRANSIENT_LOCAL="transient_local"),
           qos_profile_sensor_data=object())
    for package in ["geometry_msgs", "moai_nav_msgs", "nav_msgs", "sensor_msgs", "std_msgs", "std_srvs"]:
        module(package)
    module("geometry_msgs.msg", PoseStamped=RosPoseStamped, Twist=Twist)
    module("moai_nav_msgs.msg", Tracks=NS)
    module("nav_msgs.msg", Odometry=NS, Path=RosPath)
    module("sensor_msgs.msg", LaserScan=NS)
    module("std_msgs.msg", Bool=NS, String=NS)
    module("std_srvs.srv", SetBool=NS(Request=NS, Response=NS))
    return modules


def load_tracker(parameters=None):
    """Import the real node against minimal message/transport test doubles."""
    source = Path(__file__).resolve().parents[1] / "moai_jackal_spubert" / "safe_path_tracker_node.py"
    spec = importlib.util.spec_from_file_location("moai_jackal_spubert._tracker_test", source)
    tracker = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, ros_test_modules(parameters)):
        spec.loader.exec_module(tracker)
    return tracker.SafePathTrackerNode()


def path_message(points, stamp_ns=100_000_000_000, frame="odom"):
    return NS(
        header=NS(frame_id=frame, stamp=NS(sec=stamp_ns // 1_000_000_000, nanosec=stamp_ns % 1_000_000_000)),
        poses=[NS(pose=NS(position=NS(x=x, y=y))) for x, y in points],
    )


def ros_header(frame="odom", stamp_ns=100_000_000_000):
    return NS(frame_id=frame, stamp=NS(sec=stamp_ns // 1_000_000_000, nanosec=stamp_ns % 1_000_000_000))


def refresh_sensor_stamps(node):
    """Keep sensors current when a regression deliberately ages the plan."""
    node._odom_stamp_s = node._scan_stamp_s = node._tracks_stamp_s = node._test_now
    stamp = ros_header(stamp_ns=int(node._test_now * 1e9)).stamp
    node._odom.header.stamp = stamp
    node._scan.header.stamp = stamp
    node._tracks_source_stamp = stamp


def context_message(goal=(4.0, 0.0), goal_id="session:1", state="active", reason="", stamp_ns=100_000_000_000, frame="odom", points=None):
    return NS(data=json.dumps({
        "goal_id": goal_id, "state": state, "frame_id": frame,
        "goal": None if goal is None else list(goal), "path_stamp_ns": stamp_ns, "reason": reason,
        "path": points if points is not None else [[0, 0], [2, 0], [4, 0]],
    }))


def ready_tracker(*, parameters=None, robot=(0.0, 0.0), goal=(4.0, 0.0), points=None):
    node = load_tracker(parameters)
    node._on_odom(NS(header=ros_header("odom"), pose=NS(pose=NS(
        position=NS(x=robot[0], y=robot[1]), orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0),
    ))))
    node._on_scan(NS(header=ros_header("base_link"), ranges=[5.0], angle_min=0.0, angle_increment=0.1))
    node._on_tracks(NS(header=ros_header("base_link"), tracks=[]))
    node._on_goal(NS(header=ros_header("odom"), pose=NS(position=NS(x=goal[0], y=goal[1]))))
    node._on_path(path_message(points or [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)]))
    node._on_plan_context(context_message(goal=goal, points=points))
    node.motion_enabled = True  # Explicit test arming, never a production default.
    return node


class TrackerTests(unittest.TestCase):
    def diagnostic(self, node):
        return json.loads(node.diagnostics_pub.messages[-1].data)

    def assertStopped(self, node, reason):
        command = node.cmd_pub.messages[-1]
        self.assertEqual((command.linear.x, command.angular.z), (0.0, 0.0))
        self.assertEqual(self.diagnostic(node)["stop_reason"], reason)

    def test_defaults_are_disarmed_and_context_uses_latched_reliable_qos(self):
        node = load_tracker()
        self.assertFalse(node.motion_enabled)
        self.assertTrue(node.require_plan_context)
        for topic in [node.plan_context_topic, node.goal_completion_topic]:
            qos = node._test_qos[topic]
            self.assertEqual((qos.depth, qos.reliability, qos.durability), (1, "reliable", "transient_local"))
        node._control_tick()
        self.assertStopped(node, "disarmed")

    def test_requested_and_applied_turn_are_distinct_and_acceleration_limited(self):
        node = ready_tracker(points=[(0, 1), (0, 3)])
        node._control_tick()
        record = self.diagnostic(node)
        self.assertAlmostEqual(record["requested_angular"], 0.55)
        self.assertAlmostEqual(record["applied_angular"], 0.12)
        self.assertAlmostEqual(record["heading_error"], math.pi / 2)
        self.assertEqual(record["lookahead"], [0, 1.7])
        self.assertEqual(record["requested_linear"], 0)
        self.assertIsNone(record["stop_reason"])

    def test_opposite_requested_turn_does_not_instantly_reverse_applied_command(self):
        node = ready_tracker()
        node._last_command.angular.z = 0.55
        node._publish_command(0, -0.55)
        record = self.diagnostic(node)
        self.assertEqual(record["requested_angular"], -0.55)
        self.assertAlmostEqual(record["applied_angular"], 0.43)

    def test_approach_cap_does_not_override_rotation_or_obstacle_slowdown(self):
        node = ready_tracker(goal=(0.6, 0))
        node._control_tick()
        self.assertAlmostEqual(self.diagnostic(node)["requested_linear"], 0.07)
        node._scan.ranges = [1.0]
        node._control_tick()
        self.assertLess(self.diagnostic(node)["requested_linear"], 0.07)
        node = ready_tracker(goal=(0.6, 0), points=[(0, 1), (0, 3)])
        node._control_tick()
        self.assertEqual(self.diagnostic(node)["requested_linear"], 0)

    def test_consumed_short_plan_holds_without_turning_back_and_accepts_fresh_plan(self):
        node = ready_tracker(robot=(0.19, 0), points=[[0, 0], [0.2, 0]])
        node._control_tick()
        self.assertIsNone(self.diagnostic(node)["stop_reason"])
        for x in [0.2, 0.21]:
            node._odom.pose.pose.position.x = x
            node._control_tick()
            self.assertStopped(node, "path_exhausted")
            self.assertTrue(node.motion_enabled)
            self.assertEqual(len(node.completion_pub.messages), 0)
        node._on_plan_context(context_message(points=[[0.21, 0], [1.0, 0]]))
        node._control_tick()
        self.assertIsNone(self.diagnostic(node)["stop_reason"])
        self.assertGreater(node.cmd_pub.messages[-1].linear.x, 0)

    def test_corner_with_remaining_path_still_rotates_normally(self):
        node = ready_tracker(robot=(1, 0), points=[[0, 0], [1, 0], [1, 0], [1, 1]])
        node._control_tick()
        self.assertIsNone(self.diagnostic(node)["stop_reason"])
        self.assertEqual(node.cmd_pub.messages[-1].linear.x, 0)
        self.assertGreater(node.cmd_pub.messages[-1].angular.z, 0)

    def test_corridor_side_wall_can_slow_or_stop_existing_forward_cone(self):
        half_width = 0.6  # A 1.2 m corridor; the nominal 0.5 m disk fits centrally.
        for yaw_degrees in [0, 20]:
            with self.subTest(yaw_degrees=yaw_degrees):
                yaw = math.radians(yaw_degrees)
                rays = [yaw + bearing for bearing in [-0.7, 0, 0.7]]
                ranges = [half_width / abs(math.sin(angle)) if abs(math.sin(angle)) > 1e-12 else math.inf
                          for angle in rays]
                node = ready_tracker()
                node._odom.pose.pose.orientation.z = math.sin(yaw / 2)
                node._odom.pose.pose.orientation.w = math.cos(yaw / 2)
                node._on_scan(NS(header=ros_header("base_link"), ranges=ranges,
                                 angle_min=-0.7, angle_increment=0.7))
                node._control_tick()
                record = self.diagnostic(node)
                expected_distance = half_width / math.sin(0.7 + abs(yaw))
                self.assertAlmostEqual(record["minimum_front_distance_m"], expected_distance)
                self.assertAlmostEqual(abs(record["minimum_front_bearing_rad"]), 0.7)
                if yaw_degrees == 0:
                    self.assertIsNone(record["stop_reason"])
                    self.assertAlmostEqual(record["requested_linear"], 0.25 * (expected_distance - 0.75) / (1.40 - 0.75))
                    self.assertLess(record["requested_linear"], 0.08)
                else:
                    self.assertStopped(node, "obstacle_too_close:0.69m")
                    self.assertLess(expected_distance, 0.75)

    def test_existing_safety_stops_remain_effective(self):
        cases = [
            ("_odom", None, "odom_missing_or_stale"),
            ("_odom_stamp_s", 98.0, "odom_missing_or_stale"),
            ("_scan_stamp_s", 98.0, "scan_missing_or_stale"),
            ("_scan_frame_valid", False, "scan_frame_mismatch:got=base_link,expected=base_link"),
            ("_tracks_stamp_s", 98.0, "pedestrian_tracker_heartbeat_stale"),
            ("_tracks_frame_valid", False, "tracks_frame_mismatch:got=base_link,expected=base_link"),
            ("_minimum_human_distance", 0.2, "human_too_close:0.20m"),
            ("_path", None, "predicted_path_missing_or_stale"),
            ("_path_stamp_s", 98.0, "predicted_path_missing_or_stale"),
            ("_plan_context", None, "plan_context_missing"),
            ("_plan_context_stamp_s", 98.0, "plan_context_missing_or_stale"),
        ]
        for attribute, value, reason in cases:
            with self.subTest(reason=reason, attribute=attribute):
                node = ready_tracker()
                setattr(node, attribute, value)
                node._last_command.linear.x = 0.25
                node._last_command.angular.z = 0.55
                node._control_tick()
                self.assertStopped(node, reason)
        node = ready_tracker()
        node._scan.ranges = [0.4]
        node._control_tick()
        self.assertStopped(node, "obstacle_too_close:0.40m")

    def test_estop_latches_and_prevents_rearming(self):
        node = ready_tracker()
        node._on_emergency_stop(NS(data=True))
        self.assertStopped(node, "emergency_stop_latched")
        response = node._on_enable_motion(NS(data=True), NS())
        self.assertFalse(response.success)
        self.assertFalse(node.motion_enabled)

    def test_nonfinite_robot_position_or_orientation_stops(self):
        for component, value in [("position", math.nan), ("position", math.inf), ("orientation", math.nan)]:
            with self.subTest(component=component, value=value):
                node = ready_tracker()
                setattr(getattr(node._odom.pose.pose, component), "x", value)
                node._control_tick()
                self.assertStopped(node, "nonfinite_robot_pose")

    def test_zero_quaternion_stops_instead_of_becoming_zero_yaw(self):
        node = ready_tracker()
        node._odom.pose.pose.orientation.w = 0.0
        node._control_tick()
        self.assertStopped(node, "invalid_robot_orientation")

    def test_nonfinite_legacy_goal_stops(self):
        node = ready_tracker(parameters={"require_plan_context": False})
        node._goal.pose.position.x = math.nan
        node._control_tick()
        self.assertStopped(node, "nonfinite_goal")

    def test_nonfinite_commands_never_reach_velocity_publisher(self):
        for linear, angular in [(math.nan, 0), (0, math.nan), (math.inf, 0), (0, -math.inf)]:
            with self.subTest(linear=linear, angular=angular):
                node = ready_tracker()
                command = node._publish_command(linear, angular)
                self.assertEqual((command.linear.x, command.angular.z), (0, 0))
                self.assertStopped(node, "nonfinite_requested_command")
        node = ready_tracker()
        node._last_command.linear.x = math.nan
        node._publish_command(0.25, 0)
        self.assertStopped(node, "nonfinite_applied_command")

    def test_nonfinite_track_cannot_hide_a_nearby_person(self):
        for bad_value in [math.nan, math.inf]:
            node = ready_tracker()
            node._on_tracks(NS(header=ros_header("base_link"), tracks=[
                NS(pose=NS(position=NS(x=bad_value, y=0))),
                NS(pose=NS(position=NS(x=0.2, y=0))),
            ]))
            node._control_tick()
            self.assertStopped(node, "tracks_data_invalid")

    def test_empty_invalid_angles_or_unusable_front_scan_stops(self):
        cases = [
            ([], 0, 0.1, "scan_empty"),
            ([5.0], math.nan, 0.1, "scan_angles_invalid"),
            ([5.0], 0, math.nan, "scan_angles_invalid"),
            ([math.nan, math.nan], 0, 0.1, "scan_front_sector_invalid"),
            ([-math.inf], 0, 0.1, "scan_front_sector_invalid"),
            ([5.0], math.pi, 0.1, "scan_front_sector_invalid"),
        ]
        for ranges, angle_min, increment, reason in cases:
            with self.subTest(reason=reason, ranges=ranges):
                node = ready_tracker()
                node._on_scan(NS(header=ros_header("base_link"), ranges=ranges,
                                 angle_min=angle_min, angle_increment=increment))
                node._control_tick()
                self.assertStopped(node, reason)

    def test_positive_infinite_no_return_and_mixed_valid_scan_keep_existing_policy(self):
        for ranges in [[math.inf], [math.nan, 5.0], [math.nan, math.inf]]:
            node = ready_tracker()
            node._on_scan(NS(header=ros_header("base_link"), ranges=ranges,
                             angle_min=0, angle_increment=0.1))
            node._control_tick()
            self.assertIsNone(self.diagnostic(node)["stop_reason"])
            self.assertGreater(node.cmd_pub.messages[-1].linear.x, 0)

    def test_fresh_receipt_cannot_hide_stale_missing_or_future_sensor_header(self):
        for sensor in ["odom", "scan", "tracks"]:
            for stamp, error in [
                (ros_header(stamp_ns=95_000_000_000).stamp, "stale"),
                (ros_header(stamp_ns=101_000_000_000).stamp, "in_future"),
                (ros_header(stamp_ns=0).stamp, "missing"),
                (None, "missing"),
                (NS(sec=100, nanosec=-1), "invalid"),
            ]:
                with self.subTest(sensor=sensor, error=error):
                    node = ready_tracker()
                    if sensor == "tracks":
                        node._tracks_source_stamp = stamp
                    else:
                        getattr(node, f"_{sensor}").header.stamp = stamp
                    node._control_tick()
                    self.assertStopped(node, f"{sensor}_stamp_{error}")

    def test_sensor_header_check_has_explicit_opt_out_but_receipt_timeout_remains(self):
        node = ready_tracker(parameters={"check_sensor_header_stamps": False})
        node._odom.header.stamp = ros_header(stamp_ns=95_000_000_000).stamp
        node._control_tick()
        self.assertIsNone(self.diagnostic(node)["stop_reason"])
        node._odom_stamp_s = 95.0
        node._control_tick()
        self.assertStopped(node, "odom_missing_or_stale")

    def test_optional_tracker_does_not_require_track_source_stamp(self):
        node = ready_tracker(parameters={"require_tracks_heartbeat": False})
        node._tracks_source_stamp = None
        node._control_tick()
        self.assertIsNone(self.diagnostic(node)["stop_reason"])

    def test_atomic_context_and_visualization_path_can_arrive_in_either_order(self):
        for first in ["path", "context"]:
            with self.subTest(first=first):
                node = ready_tracker()
                new_path = path_message([(0, 0), (-2, 0)], stamp_ns=100_200_000_000)
                new_context = context_message(stamp_ns=100_100_000_000, points=[[0, 0], [3, 0]])
                callbacks = [(node._on_path, new_path), (node._on_plan_context, new_context)]
                if first == "context":
                    callbacks.reverse()
                callbacks[0][0](callbacks[0][1])
                node._control_tick()
                self.assertIsNone(self.diagnostic(node)["stop_reason"])
                callbacks[1][0](callbacks[1][1])
                node._control_tick()
                self.assertIsNone(self.diagnostic(node)["stop_reason"])
                self.assertEqual(node._path.poses[-1].pose.position.x, 3)
                self.assertEqual(node._path_stamp_ns(node._path), 100_100_000_000)

    def test_empty_or_fresh_visualization_path_neither_stops_nor_refreshes_execution(self):
        node = ready_tracker()
        received_s = node._path_stamp_s
        node._on_path(path_message([]))
        node._control_tick()
        self.assertIsNone(self.diagnostic(node)["stop_reason"])
        node._test_now += 1.3
        refresh_sensor_stamps(node)
        node._on_path(path_message([(0, 0), (3, 0)], stamp_ns=101_300_000_000))
        self.assertEqual(node._path_stamp_s, received_s)
        node._control_tick()
        self.assertStopped(node, "plan_context_missing_or_stale")

    def test_late_durable_context_cannot_execute_after_receipt_and_arming(self):
        node = ready_tracker()
        node.motion_enabled = False
        node._test_now = 105.0
        refresh_sensor_stamps(node)
        node._on_plan_context(context_message(stamp_ns=100_000_000_000))
        self.assertEqual(node._path_stamp_s, 105.0)
        self.assertTrue(node._on_enable_motion(NS(data=True), NS()).success)
        node._control_tick()
        self.assertStopped(node, "plan_stamp_stale")
        self.assertAlmostEqual(self.diagnostic(node)["path_header_age_sec"], 5.0)

    def test_far_future_plan_stamp_fails_closed(self):
        node = ready_tracker()
        node._on_plan_context(context_message(stamp_ns=100_500_000_000))
        node._control_tick()
        self.assertStopped(node, "plan_stamp_in_future")

    def test_context_goal_is_used_instead_of_stale_visualization_goal(self):
        node = ready_tracker(goal=(4, 0))
        node._on_plan_context(context_message(goal=(0.4, 0)))
        node._control_tick()
        self.assertStopped(node, "goal_reached_disarmed")
        self.assertFalse(node.motion_enabled)
        self.assertEqual(json.loads(node.completion_pub.messages[-1].data), {"goal_id": "session:1"})

    def test_late_active_or_path_cannot_restart_a_completed_goal(self):
        node = ready_tracker(goal=(0.4, 0))
        node._control_tick()
        node._on_path(path_message([(0, 0), (4, 0)]))
        node._on_plan_context(context_message(goal=(4, 0)))
        self.assertFalse(node._on_enable_motion(NS(data=True), NS()).success)
        node._control_tick()
        self.assertStopped(node, "goal_reached_disarmed")
        self.assertEqual(len(node.completion_pub.messages), 1)

    def test_bridge_completion_disarms_immediately_and_new_goal_does_not_arm(self):
        node = ready_tracker()
        node._on_plan_context(context_message(state="complete", goal=None, stamp_ns=None))
        self.assertStopped(node, "goal_reached_disarmed")
        self.assertFalse(node.motion_enabled)
        node._on_plan_context(context_message(goal_id="session:2", state="waiting", goal=None, stamp_ns=None, reason="new_goal"))
        self.assertFalse(node.motion_enabled)
        self.assertIsNone(node._completed_goal_id)
        self.assertTrue(node._on_enable_motion(NS(data=True), NS()).success)
        node._control_tick()
        self.assertStopped(node, "hold:new_goal")

    def test_hold_reason_survives_empty_path_and_control_tick(self):
        node = ready_tracker()
        node._on_plan_context(context_message(state="hold", reason="robot_footprint_collision", stamp_ns=None))
        node._on_path(path_message([]))
        node._control_tick()
        self.assertStopped(node, "hold:robot_footprint_collision")

    def test_waiting_and_complete_context_may_omit_geometry(self):
        node = ready_tracker()
        node._on_plan_context(NS(data=json.dumps({"goal_id": "session:2", "state": "waiting", "reason": "new_goal"})))
        self.assertStopped(node, "hold:new_goal")
        self.assertIsNone(node._path)
        node._on_plan_context(NS(data=json.dumps({"goal_id": "session:2", "state": "complete"})))
        self.assertStopped(node, "goal_reached_disarmed")
        self.assertFalse(node.motion_enabled)

    def test_malformed_context_and_frame_mismatch_fail_closed(self):
        malformed = ["[]", "{}", '{"goal_id":"x","state":"active","frame_id":"odom","goal":[NaN,0],"path_stamp_ns":1}']
        for points in [[], [[0, 0]], [[0, 0], [1, math.inf]], [[0, 0], [1]]]:
            malformed.append(context_message(points=points).data)
        malformed.append(context_message(stamp_ns=(2 ** 31) * 1_000_000_000).data)
        for payload in malformed:
            node = ready_tracker()
            node._on_plan_context(NS(data=payload))
            self.assertStopped(node, "plan_context_invalid")
            self.assertIsNone(node._path)
        node = ready_tracker()
        node._on_plan_context(context_message(frame="map"))
        node._control_tick()
        self.assertStopped(node, "frame_mismatch:path=map,goal=map,odom=odom")

    def test_explicit_legacy_mode_still_requires_goal_and_fresh_path(self):
        node = ready_tracker(parameters={"require_plan_context": False})
        node._plan_context = None
        node._goal = None
        node._control_tick()
        self.assertStopped(node, "final_goal_missing")

    def test_jsonl_contains_same_applied_command_as_topic(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "tracker.jsonl"
            node = ready_tracker(parameters={"diagnostics_path": str(output)})
            node._control_tick()
            record = json.loads(output.read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual(record, self.diagnostic(node))
            self.assertAlmostEqual(record["applied_linear"], node.cmd_pub.messages[-1].linear.x)
            node.destroy_node()


if __name__ == "__main__":
    unittest.main()
