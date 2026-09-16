"""Exercise the real bridge callbacks with deterministic inference/ROS doubles."""
import importlib.util
import io
import json
import math
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

from test_tracker_stability import ready_tracker, ros_test_modules
from moai_jackal_spubert.guided_spubert_runtime import GuidedInferenceResult


class StampTime:
    def __init__(self, nanoseconds=0, *, seconds=0):
        self.nanoseconds = int(nanoseconds + seconds * 1e9)

    def to_msg(self):
        return NS(sec=self.nanoseconds // 1_000_000_000, nanosec=self.nanoseconds % 1_000_000_000)

    def __add__(self, other):
        return StampTime(self.nanoseconds + other.nanoseconds)


class PoseStamped:
    def __init__(self):
        self.header = NS(frame_id="", stamp=StampTime().to_msg())
        self.pose = NS(position=NS(x=0.0, y=0.0, z=0.0), orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0))


class PathMessage:
    def __init__(self):
        self.header = NS(frame_id="", stamp=StampTime().to_msg())
        self.poses = []


class ManualExecutor:
    """Deterministic worker: submit never runs model work on the callback."""
    def __init__(self, **kwargs):
        self.jobs = []

    def submit(self, function, *args):
        future = Future()
        self.jobs.append((future, function, args))
        return future

    def finish(self):
        future, function, args = self.jobs.pop(0)
        try:
            future.set_result(function(*args))
        except Exception as exc:
            future.set_exception(exc)

    def shutdown(self, **kwargs):
        pass


def plan_tick(node):
    """Finish a zero-latency plan for ordinary callback regression tests."""
    node._on_timer()
    if node._pending_inference is not None:
        node._inference_executor.finish()
        node._on_timer()


def fresh_inputs(node, now, *, obstacle=False, humans=()):
    node._test_now = now
    stamp = StampTime(seconds=now).to_msg()
    node._latest_odom.header.stamp = stamp
    node._on_odom(node._latest_odom)
    scan = node._latest_scan
    scan.header.stamp = stamp
    if obstacle:
        scan.ranges = [math.inf] * 720
        scan.ranges[360] = 0.8
    node._on_scan(scan)
    node._on_tracks(NS(header=NS(frame_id="base_link", stamp=stamp), tracks=[
        NS(score=1.0, id=index + 1, pose=NS(position=NS(x=x, y=y)))
        for index, (x, y) in enumerate(humans)
    ]))


def candidate(*, valid=True, nan=False):
    points = [(0.1 * step, 0.0) for step in range(1, 13)]
    if nan:
        points[5] = (math.nan, 0.0)
    return GuidedInferenceResult(points, [(1.2, 0)], (1.2, 0), (8, 0), valid, True, True, guidance_distance_m=6.8)


def load_bridge():
    modules = ros_test_modules({"diagnostics_path": "", "use_cuda": False})
    node_class = modules["rclpy.node"].Node
    node_class.get_clock = lambda self: NS(now=lambda: StampTime(int(self._test_now * 1e9)))
    node_class.get_logger = lambda self: NS(info=self._test_log.append, warning=lambda *a, **k: None, error=self._test_log.append)
    node_class.create_timer = lambda self, period, callback: self.__dict__.setdefault("_test_timers", []).append((period, callback))
    modules["geometry_msgs.msg"].PoseStamped = PoseStamped
    modules["geometry_msgs.msg"].Point = NS
    modules["nav_msgs.msg"].Path = PathMessage
    modules["nav_msgs.msg"].OccupancyGrid = NS
    modules["std_msgs.msg"].ColorRGBA = NS
    modules["rclpy.qos"].QoSDurabilityPolicy = modules["rclpy.qos"].DurabilityPolicy
    modules["rclpy.qos"].QoSReliabilityPolicy = modules["rclpy.qos"].ReliabilityPolicy
    for name, attrs in {
        "rclpy.action": {"ActionClient": lambda *a: NS(wait_for_server=lambda **k: False)},
        "rclpy.duration": {"Duration": StampTime},
        "rclpy.time": {"Time": StampTime},
        "nav2_msgs": {}, "nav2_msgs.action": {"ComputePathToPose": NS(Goal=NS)},
        "tf2_ros": {"Buffer": NS, "TransformException": RuntimeError, "TransformListener": lambda *a: None},
        "visualization_msgs": {}, "visualization_msgs.msg": {"Marker": NS, "MarkerArray": NS},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        modules[name] = module
    source = Path(__file__).resolve().parents[1] / "moai_jackal_spubert" / "real_jackal_spubert_bridge_node.py"
    spec = importlib.util.spec_from_file_location("moai_jackal_spubert._bridge_test", source)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)

    class Runtime:
        obs_len, pred_len = 8, 12

        def __init__(self, **kw):
            self.last_input_diagnostics = {}
            self.candidates = [candidate()]
            self.calls = []
            self.delay = 0.0

        def predict_candidates(self, **kw):
            self.calls.append(kw)
            self.last_input_diagnostics = {"theta_rad": kw["heading_override_rad"]}
            node._test_now += self.delay
            return self.candidates

    module.GuidedSpubertRuntime = Runtime
    module.ThreadPoolExecutor = ManualExecutor
    node = module.RealJackalSpubertBridgeNode()
    node._publish_debug_map = lambda *a: None
    node._publish_markers = lambda *a: None
    node._route_in_frame = lambda frame: [(0, 0), (8, 0)]
    node._diagnostics_file = io.StringIO()
    return node


def ready_bridge():
    node = load_bridge()
    for index in range(8):
        node._test_now = 97.2 + index * 0.4
        odom = NS(header=NS(frame_id="odom", stamp=StampTime(int(node._test_now * 1e9)).to_msg()), pose=NS(pose=PoseStamped().pose))
        node._on_odom(odom)
        node._on_tracks(NS(header=NS(frame_id="base_link", stamp=StampTime(int(node._test_now * 1e9)).to_msg()), tracks=[]))
        node._sample_histories()
    node._on_scan(NS(header=NS(frame_id="base_link", stamp=StampTime(int(node._test_now * 1e9)).to_msg()), ranges=[math.inf] * 720,
                     angle_min=-math.pi, angle_increment=2 * math.pi / 720, range_min=0.35, range_max=30.0))
    goal = PoseStamped()
    goal.header.frame_id = "odom"
    goal.pose.position.x = 8.0
    node._on_goal(goal)
    return node


class BridgeTests(unittest.TestCase):
    def context(self, node):
        return json.loads(node.plan_context_pub.messages[-1].data)

    def records(self, node):
        return [json.loads(line) for line in node._diagnostics_file.getvalue().splitlines()]

    def test_valid_plan_contains_atomic_goal_and_path_and_actual_heading(self):
        node = ready_bridge()
        node._robot_history[-1] = (0, 0.0011)
        plan_tick(node)
        context = self.context(node)
        self.assertEqual(context["state"], "active")
        self.assertEqual(context["goal"], [8, 0])
        self.assertEqual(len(context["path"]), 12)
        self.assertEqual(context["path_stamp_ns"], node._stamp_ns(node.path_pub.messages[-1].header.stamp))
        self.assertEqual(node._runtime.calls[-1]["heading_override_rad"], 0)
        record = self.records(node)[-1]
        self.assertEqual(record["model_input"]["theta_rad"], 0)
        self.assertAlmostEqual(record["heading"]["legacy_heading_rad"], math.pi / 2)
        self.assertEqual(len(record["history_sample_times_s"]), 8)

    def test_all_rejected_does_not_republish_previous_path(self):
        node = ready_bridge()
        node._previous_path = [(0, 0), (3, 0)]
        node._runtime.candidates = [candidate(valid=False)]
        plan_tick(node)
        self.assertEqual(self.context(node)["state"], "hold")
        self.assertEqual(node.path_pub.messages[-1].poses, [])
        self.assertIsNone(node._previous_path)
        self.assertFalse(self.records(node)[-1]["valid"])

    def test_nonfinite_prediction_is_recorded_and_stopped(self):
        node = ready_bridge()
        node._runtime.candidates = [candidate(nan=True)]
        plan_tick(node)
        record = self.records(node)[-1]
        self.assertFalse(record["valid"])
        self.assertIsNone(record["attempts"][0]["path"][5][0])
        self.assertEqual(node.path_pub.messages[-1].poses, [])

    def test_inference_delay_does_not_refresh_old_inputs(self):
        node = ready_bridge()
        node._runtime.delay = 0.61
        plan_tick(node)
        self.assertEqual(self.context(node)["reason"], "inputs_stale_after_inference")
        self.assertEqual(node.path_pub.messages[-1].poses, [])

    def test_worker_longer_than_sensor_timeout_accepts_only_fresh_revalidated_inputs(self):
        node = ready_bridge()
        node._on_timer()
        self.assertEqual(node._runtime.calls, [])  # callback returned before work
        self.assertIsNotNone(node._pending_inference)
        fresh_inputs(node, 100.7)
        node._sample_histories()
        node._inference_executor.finish()
        node._on_timer()
        self.assertEqual(self.context(node)["state"], "active")
        record = self.records(node)[-1]
        self.assertAlmostEqual(record["inference_result_age_sec"], 0.7)
        self.assertEqual(record["model_map_generation"], 1)
        self.assertEqual(record["validation"]["map_generation"], 2)
        self.assertEqual(record["odom_stamp_ns"], 100_000_000_000)
        self.assertEqual(record["validation"]["odom_stamp_ns"], 100_700_000_000)
        self.assertNotEqual(record["history_sample_times_s"], record["validation"]["history_sample_times_s"])
        self.assertTrue(record["model_output_collected"])

    def test_one_second_inference_restarts_without_extra_poll_delay(self):
        node = ready_bridge()
        periods = [period for period, callback in node._test_timers if callback == node._on_timer]
        self.assertEqual(periods, [0.05])
        node._on_timer()
        publication_times = []
        for now in (101.0, 102.0, 103.0):
            fresh_inputs(node, now)
            node._inference_executor.finish()
            node._on_timer()
            self.assertEqual(self.context(node)["state"], "active")
            publication_times.append(self.context(node)["path_stamp_ns"] * 1e-9)
            self.assertIsNotNone(node._pending_inference)
            self.assertAlmostEqual(node._pending_inference[1]["start_s"], now)
            self.assertEqual(len(node._inference_executor.jobs), 1)
        self.assertTrue(all(b - a < 1.2 for a, b in zip(publication_times, publication_times[1:])))

    def test_new_obstacle_during_inference_is_checked_on_new_map(self):
        node = ready_bridge()
        node._on_timer()
        snapshot = node._inference_executor.jobs[0][2][0]
        self.assertIsNot(snapshot, node.map_provider)
        self.assertFalse(snapshot.grid is node.map_provider.grid)
        fresh_inputs(node, 100.7, obstacle=True)
        node._inference_executor.finish()
        node._on_timer()
        self.assertEqual(self.context(node)["state"], "hold")
        record = self.records(node)[-1]
        self.assertEqual(record["attempts"][0]["reason"], "robot_footprint_collision")
        self.assertIsNotNone(record["attempts"][0]["first_collision"])
        self.assertEqual(snapshot.path_collision_cost([(0.8, 0)], radius=0.5, weight=1), 0)
        self.assertGreater(node.map_provider.path_collision_cost([(0.8, 0)], radius=0.5, weight=1), 0)

    def test_new_person_during_inference_is_not_lost_before_history_timer(self):
        node = ready_bridge()
        node._on_timer()
        fresh_inputs(node, 100.7, humans=[(0.6, 0)])
        self.assertEqual(node._human_histories, {})
        node._inference_executor.finish()
        node._on_timer()
        self.assertEqual(self.context(node)["state"], "hold")
        self.assertEqual(self.records(node)[-1]["attempts"][0]["reason"], "predicted_human_clearance")

    def test_current_human_anchor_preserves_sampled_velocity(self):
        node = ready_bridge()
        node._human_histories = {1: [(0, 0), (0.2, 0), (0.4, 0)]}
        node._latest_humans = {1: (0.5, 0), 2: (4, 1)}
        current = node._current_human_histories()
        self.assertEqual(current[1][-1], (0.5, 0))
        self.assertAlmostEqual(current[1][-1][0] - current[1][0][0], 0.4)
        self.assertEqual(current[2], [(4, 1)])

    def test_goal_changed_while_pending_discards_old_model_result(self):
        node = ready_bridge()
        node._on_timer()
        old_id = node._mission.goal_id
        node._on_goal(node._goal)
        node._inference_executor.finish()
        node._on_timer()
        self.assertNotEqual(node._mission.goal_id, old_id)
        self.assertEqual(self.context(node)["reason"], "inference_goal_changed")
        self.assertEqual(self.context(node)["state"], "hold")
        self.assertEqual(node.path_pub.messages[-1].poses, [])

    def test_pose_frame_changed_while_pending_discards_old_model_result(self):
        node = ready_bridge()
        node._on_timer()
        node._latest_odom.header.frame_id = "new_odom"
        fresh_inputs(node, 100.2)
        node._inference_executor.finish()
        node._on_timer()
        self.assertEqual(self.context(node)["reason"], "inference_pose_frame_changed")

    def test_frame_switch_back_still_discards_result_from_before_reset(self):
        node = ready_bridge()
        node._on_timer()
        for frame, now in [("temporary_odom", 100.1), ("odom", 100.2)]:
            node._latest_odom.header.frame_id = frame
            fresh_inputs(node, now)
        node._inference_executor.finish()
        node._on_timer()
        self.assertEqual(self.context(node)["reason"], "inference_pose_frame_changed")

    def test_goal_completed_while_worker_pending_cannot_reactivate(self):
        node = ready_bridge()
        node._on_timer()
        node._on_goal_completion(NS(data=json.dumps({"goal_id": node._mission.goal_id})))
        node._inference_executor.finish()
        node._on_timer()
        self.assertEqual(self.context(node)["state"], "complete")
        self.assertEqual(node.path_pub.messages[-1].poses, [])

    def test_slow_preprocessing_does_not_dispatch_stale_input_to_worker(self):
        node = ready_bridge()
        node._publish_debug_map = lambda *a: setattr(node, "_test_now", node._test_now + 0.61)
        node._on_timer()
        self.assertIsNone(node._pending_inference)
        self.assertEqual(node._runtime.calls, [])
        self.assertEqual(self.context(node)["reason"], "inputs_stale_before_inference")

    def test_worker_result_expiry_does_not_extend_sensor_or_plan_timeouts(self):
        node = ready_bridge()
        node._on_timer()
        fresh_inputs(node, 101.21)
        node._inference_executor.finish()
        node._on_timer()
        self.assertEqual(self.context(node)["reason"], "inference_result_expired")
        self.assertEqual(node.path_pub.messages[-1].poses, [])
        self.assertFalse(self.records(node)[-1]["model_output_collected"])

    def test_pending_timeout_is_recorded_once_without_launching_another_worker(self):
        node = ready_bridge()
        node._on_timer()
        fresh_inputs(node, 101.21)
        node._on_timer()
        node._on_timer()
        self.assertEqual(self.context(node)["reason"], "inference_result_expired")
        self.assertEqual(len(self.records(node)), 1)
        self.assertEqual(len(node._inference_executor.jobs), 1)

    def test_model_age_expiry_after_validation_is_not_mislabeled_sensor_stale(self):
        node = ready_bridge()
        node._on_timer()
        fresh_inputs(node, 100.95)
        node._inference_executor.finish()
        node._publish_markers = lambda *a: setattr(node, "_test_now", node._test_now + 0.26)
        node._on_timer()
        self.assertEqual(self.context(node)["reason"], "inference_result_expired_after_validation")
        self.assertFalse(node._sensor_stamp_error())

    def test_real_worker_keeps_callbacks_and_history_running_while_model_blocks(self):
        node = ready_bridge()
        node._inference_executor = ThreadPoolExecutor(max_workers=1)
        entered, release = threading.Event(), threading.Event()
        original = node._runtime.predict_candidates

        def blocked_model(**kwargs):
            entered.set()
            if not release.wait(timeout=3):
                raise TimeoutError("test failed to release model")
            return original(**kwargs)

        node._runtime.predict_candidates = blocked_model
        try:
            node._on_timer()
            self.assertTrue(entered.wait(timeout=1))
            self.assertFalse(node._pending_inference[0].done())
            fresh_inputs(node, 100.7)
            node._sample_histories()
            self.assertAlmostEqual(node._robot_history_times[-1], 100.7)
            node._on_timer()  # no re-entrant model submission
            release.set()
            node._pending_inference[0].result(timeout=2)
            node._on_timer()
            self.assertEqual(self.context(node)["state"], "active")
        finally:
            release.set()
            node._inference_executor.shutdown(wait=True)

    def test_diagnostic_disk_error_cannot_prevent_rejected_path_hold(self):
        class FullDisk:
            def write(self, _):
                raise OSError("disk full")
            def close(self):
                pass
        node = ready_bridge()
        plan_tick(node)
        self.assertEqual(self.context(node)["state"], "active")
        node._last_predict_s = -math.inf
        node._runtime.candidates = [candidate(valid=False)]
        node._diagnostics_file = FullDisk()
        plan_tick(node)
        self.assertEqual(self.context(node)["state"], "hold")
        self.assertIsNone(node._diagnostics_file)
        self.assertEqual(node.path_pub.messages[-1].poses, [])

    def test_validation_delay_also_rejects_before_publication(self):
        node = ready_bridge()
        node._publish_markers = lambda *a: setattr(node, "_test_now", node._test_now + 0.61)
        plan_tick(node)
        self.assertEqual(self.context(node)["reason"], "inputs_stale_after_validation")
        self.assertEqual(node.path_pub.messages[-1].poses, [])

    def test_completed_goal_stays_complete_when_pose_or_goal_transform_moves(self):
        node = ready_bridge()
        goal_id = node._mission.goal_id
        node._on_goal_completion(NS(data=json.dumps({"goal_id": goal_id})))
        plan_tick(node)
        self.assertEqual(self.context(node)["state"], "complete")
        self.assertEqual(node._runtime.calls, [])
        node._on_goal(node._goal)
        self.assertFalse(node._mission.completed)
        node._on_goal_completion(NS(data=json.dumps({"goal_id": goal_id})))
        self.assertFalse(node._mission.completed)

    def test_bridge_distance_completion_is_shared_without_a_new_path(self):
        node = ready_bridge()
        node._goal.pose.position.x = 0.4
        plan_tick(node)
        self.assertEqual(self.context(node)["state"], "complete")
        self.assertEqual(node._runtime.calls, [])

    def test_frame_change_clears_histories_and_old_human_positions(self):
        node = ready_bridge()
        node._latest_humans = {1: (2, 3)}
        odom = NS(header=NS(frame_id="different_odom", stamp=StampTime().to_msg()), pose=node._latest_odom.pose)
        node._on_odom(odom)
        self.assertEqual(list(node._robot_history), [])
        self.assertEqual(node._latest_humans, {})
        self.assertFalse(node._tracks_frame_valid)

    def test_no_goal_does_not_publish_an_invalid_empty_mission(self):
        node = load_bridge()
        node._hold("waiting_for_goal", publish_empty=False)
        self.assertEqual(node.plan_context_pub.messages, [])

    def test_nonfinite_robot_and_empty_orientation_never_infer(self):
        for value, expected in [(math.nan, "nonfinite_robot_pose"), (0.0, "invalid_robot_orientation")]:
            node = ready_bridge()
            node._latest_odom.pose.pose.orientation.w = value
            plan_tick(node)
            self.assertEqual(self.context(node)["reason"], expected)
            self.assertEqual(node._runtime.calls, [])

    def test_invalid_scan_is_not_treated_as_an_empty_free_map(self):
        for ranges, angle in [([], 0.0), ([math.nan] * 3, 0.0), ([math.inf] * 3, math.nan)]:
            node = ready_bridge()
            scan = node._latest_scan
            scan.ranges, scan.angle_min = ranges, angle
            node._on_scan(scan)
            plan_tick(node)
            self.assertEqual(self.context(node)["reason"], "invalid_scan_geometry_or_ranges")
            self.assertEqual(node._runtime.calls, [])

    def test_nonfinite_person_does_not_poison_or_disappear_from_input(self):
        node = ready_bridge()
        node._on_tracks(NS(header=NS(frame_id="base_link"), tracks=[
            NS(score=1.0, id=1, pose=NS(position=NS(x=math.nan, y=0.0))),
            NS(score=1.0, id=2, pose=NS(position=NS(x=0.2, y=0.0))),
        ]))
        plan_tick(node)
        self.assertEqual(self.context(node)["reason"], "nonfinite_pedestrian_track")
        self.assertEqual(node._runtime.calls, [])

    def test_old_sensor_timestamp_is_not_refreshed_by_a_fresh_callback(self):
        for name in ("odom", "scan", "tracks"):
            node = ready_bridge()
            stale = StampTime(seconds=95).to_msg()
            if name == "tracks":
                node._tracks_source_stamp = stale
            else:
                getattr(node, "_latest_" + name).header.stamp = stale
            plan_tick(node)
            self.assertEqual(self.context(node)["reason"], name + "_stamp_stale")
            self.assertEqual(node._runtime.calls, [])

    def test_actual_bridge_context_drives_tracker_and_rejection_stops_it(self):
        bridge = ready_bridge()
        tracker = ready_tracker()
        plan_tick(bridge)
        tracker._on_plan_context(bridge.plan_context_pub.messages[-1])
        tracker._control_tick()
        self.assertGreater(tracker.cmd_pub.messages[-1].linear.x, 0)
        bridge._last_predict_s = -math.inf
        bridge._runtime.candidates = [candidate(valid=False)]
        plan_tick(bridge)
        tracker._on_plan_context(bridge.plan_context_pub.messages[-1])
        tracker._control_tick()
        self.assertEqual(tracker.cmd_pub.messages[-1].linear.x, 0)
        self.assertEqual(tracker.cmd_pub.messages[-1].angular.z, 0)

    def test_tracker_completion_message_latches_actual_bridge_goal(self):
        bridge = ready_bridge()
        tracker = ready_tracker(robot=(7.6, 0))
        plan_tick(bridge)
        tracker._on_plan_context(bridge.plan_context_pub.messages[-1])
        tracker._control_tick()
        self.assertFalse(tracker.motion_enabled)
        bridge._on_goal_completion(tracker.completion_pub.messages[-1])
        self.assertTrue(bridge._mission.completed)
        self.assertEqual(self.context(bridge)["state"], "complete")


if __name__ == "__main__":
    unittest.main()
