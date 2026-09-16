#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import time
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from dataclasses import asdict
from typing import Dict, Optional, Sequence, Tuple

import rclpy
from geometry_msgs.msg import Point, PoseStamped
from moai_nav_msgs.msg import Tracks
from nav2_msgs.action import ComputePathToPose
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import ColorRGBA, String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from .navigation_core import (
    same_frame_id,
    transform_xy,
    validate_candidate_path,
    yaw_from_quaternion,
)
from .guided_spubert_runtime import (
    GuidedInferenceResult,
    GuidedSpubertRuntime,
    adaptive_guidance_point_along_path,
    sample_polyline,
)
from .rolling_laser_map import RollingLaserMapProvider
from .heading_stability import HeadingConfig, HeadingSelector
from .planning_stability import PathOption, SelectionConfig, rank_valid_paths
from .goal_lifecycle import GoalLifecycle
from .diagnostics_utils import json_finite
from .sensor_freshness import stamp_error


XY = Tuple[float, float]


class RealJackalSpubertBridgeNode(Node):
    """Adapt the real Jackal sensor topics to the guided SPU-BERT runtime.

    This node only publishes paths. Wheel commands are deliberately handled by
    the separately armed safety tracker.
    """

    def __init__(self) -> None:
        super().__init__("real_jackal_spubert_bridge")

        self.model_repo_path = self._string_param("model_repo_path", "/root/moai_social_bert_refactored")
        self.model_config_path = self._string_param(
            "model_config_path",
            "/root/moai_social_bert_refactored/configs/spubert/"
            "moai_social_nav_route_gp_continue_b21_col01_social005.yaml",
        )
        self.model_checkpoint_path = self._string_param(
            "model_checkpoint_path",
            "/root/moai_social_bert_refactored/output/"
            "spubert_route_gp_continue_b21_col01_social005/model_best.pth",
        )
        self.odom_topic = self._string_param("odom_topic", "/aft_mapped_to_init")
        self.tracks_topic = self._string_param("tracks_topic", "/ped_tracking")
        self.scan_topic = self._string_param("scan_topic", "/scan")
        self.goal_topic = self._string_param("goal_topic", "/goal_pose")
        self.path_topic = self._string_param("path_topic", "/spu_bert/predicted_path")
        self.final_goal_topic = self._string_param("final_goal_topic", "/spu_bert/final_goal_odom")
        self.global_path_topic = self._string_param("global_path_topic", "/spu_bert/global_path_odom")
        self.status_topic = self._string_param("status_topic", "/spu_bert/runtime_status")
        self.marker_topic = self._string_param("marker_topic", "/spu_bert/runtime_markers")
        self.plan_context_topic = self._string_param("plan_context_topic", "/spu_bert/plan_context")
        self.goal_completion_topic = self._string_param("goal_completion_topic", "/spu_bert/goal_completion")
        self.compute_path_action = self._string_param("compute_path_action", "/compute_path_to_pose")
        self.planner_id = self._string_param("planner_id", "")
        self.base_frame = self._string_param("base_frame", "base_link")
        self.diagnostics_path = os.path.expanduser(
            self._string_param("diagnostics_path", "/tmp/spubert_real_jackal_diagnostics.jsonl")
        )

        self.use_cuda = self._bool_param("use_cuda", True)
        self.require_global_path = self._bool_param("require_global_path", True)
        self.tracks_are_robot_relative = self._bool_param("tracks_are_robot_relative", True)
        self.use_adaptive_guidance = self._bool_param("adaptive_guidance", True)
        self.obs_len = int(self.declare_parameter("obs_len", 8).value)
        self.pred_len = int(self.declare_parameter("pred_len", 12).value)
        self.prediction_dt = float(self.declare_parameter("prediction_dt", 0.4).value)
        self.replan_period = float(self.declare_parameter("replan_period", 0.8).value)
        self.route_refresh_period = max(
            float(self.declare_parameter("route_refresh_period_sec", 2.0).value),
            0.0,
        )
        self.guidance_radius = float(self.declare_parameter("guidance_radius", 8.0).value)
        self.guidance_min_distance = float(
            self.declare_parameter("adaptive_guidance_min_distance", 2.0).value
        )
        self.guidance_probe_step = float(
            self.declare_parameter("adaptive_guidance_probe_step", 0.5).value
        )
        self.tgp_top_k = max(int(self.declare_parameter("tgp_top_k", 5).value), 1)
        self.d_sample = max(int(self.declare_parameter("d_sample", 40).value), 20)
        self.runtime_seed = int(self.declare_parameter("runtime_seed", 21).value)
        self.heading_selector = HeadingSelector(HeadingConfig(
            mode=self._string_param("model_heading_mode", "motion_guarded"),
        ))
        self.inference_max_age = float(self.declare_parameter("inference_max_age_sec", 1.20).value)
        if not math.isfinite(self.inference_max_age) or not 0 < self.inference_max_age <= 1.20:
            raise ValueError("inference_max_age_sec must be positive and at most the 1.20s plan lifetime")
        self.selection_config = SelectionConfig(
            mode=self._string_param("candidate_selection_mode", "continuous"),
            lookahead_m=float(self.declare_parameter("selection_lookahead_m", 0.70).value),
            guidance_weight=float(self.declare_parameter("selection_guidance_weight", 0.20).value),
            heading_weight=float(self.declare_parameter("selection_heading_weight", 0.50).value),
            turning_weight=float(self.declare_parameter("selection_turning_weight", 0.15).value),
            continuity_weight=float(self.declare_parameter("selection_continuity_weight", 2.00).value),
        )
        self.continuity_reference_timeout = float(
            self.declare_parameter("continuity_reference_timeout_sec", 1.20).value
        )
        if not math.isfinite(self.continuity_reference_timeout) or self.continuity_reference_timeout <= 0:
            raise ValueError("continuity_reference_timeout_sec must be positive and finite")

        self.robot_radius = float(self.declare_parameter("robot_radius", 0.34).value)
        self.static_safety_margin = float(
            self.declare_parameter("static_safety_margin", 0.16).value
        )
        self.minimum_human_center_distance = float(
            self.declare_parameter("minimum_human_center_distance", 1.20).value
        )
        self.human_radius = float(self.declare_parameter("human_radius", 0.35).value)
        self.human_safety_margin = float(
            self.declare_parameter("human_safety_margin", 0.20).value
        )
        self.maximum_model_speed = float(
            self.declare_parameter("maximum_model_speed", 1.50).value
        )
        self.maximum_step_ratio = float(
            self.declare_parameter("maximum_step_ratio", 1.50).value
        )
        self.minimum_goal_progress = float(
            self.declare_parameter("minimum_goal_progress", 0.10).value
        )
        self.goal_tolerance = float(self.declare_parameter("goal_tolerance", 0.50).value)
        self.odom_timeout = float(self.declare_parameter("odom_timeout_sec", 0.60).value)
        self.scan_timeout = float(self.declare_parameter("scan_timeout_sec", 0.60).value)
        self.tracks_timeout = float(self.declare_parameter("tracks_timeout_sec", 1.00).value)
        self.check_sensor_header_stamps = self._bool_param("check_sensor_header_stamps", True)
        self.sensor_stamp_future_tolerance = float(
            self.declare_parameter("sensor_stamp_future_tolerance_sec", 0.10).value
        )
        self.minimum_track_score = float(
            self.declare_parameter("minimum_track_score", 0.30).value
        )

        map_size = float(self.declare_parameter("rolling_map_size_m", 24.0).value)
        map_resolution = float(self.declare_parameter("rolling_map_resolution", 0.10).value)
        free_gap_fill = float(self.declare_parameter("free_gap_fill_m", 0.20).value)
        self.free_ray_limit = float(self.declare_parameter("free_ray_limit_m", 11.5).value)
        self.map_provider = RollingLaserMapProvider(
            size_m=map_size,
            resolution=map_resolution,
            free_gap_fill_m=free_gap_fill,
            unknown_is_occupied=True,
        )

        self._latest_odom: Optional[Odometry] = None
        self._latest_scan: Optional[LaserScan] = None
        self._scan_frame_valid = False
        self._tracks_frame_valid = False
        self._scan_data_error = ""
        self._tracks_data_error = ""
        self._pose_error = ""
        self._latest_scan_frame = ""
        self._latest_tracks_frame = ""
        self._latest_humans: Dict[int, XY] = {}
        self._robot_history = deque(maxlen=self.obs_len)
        self._robot_history_times = deque(maxlen=self.obs_len)
        self._robot_history_odom_stamps = deque(maxlen=self.obs_len)
        self._human_histories: Dict[int, deque[XY]] = {}
        self._odom_stamp_s = -math.inf
        self._scan_stamp_s = -math.inf
        self._tracks_stamp_s = -math.inf
        self._tracks_source_stamp = None
        self._goal: Optional[PoseStamped] = None
        self._goal_generation = 0
        self._mission = GoalLifecycle()
        self._goal_xy = None
        self._goal_frame = ""
        self._previous_path = None
        self._previous_path_stamp_s = -math.inf
        self._previous_path_frame = ""
        self._last_pose_frame = ""
        self._pose_frame_generation = 0
        self._route_generation = -1
        self._route_message: Optional[Path] = None
        self._route_pending = False
        self._last_route_request_s = -math.inf
        self._last_predict_s = -math.inf
        self._map_generation = 0
        self._pending_inference = None
        self._last_status = ""
        self._diagnostics_file = None

        if self.diagnostics_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.diagnostics_path)), exist_ok=True)
            self._diagnostics_file = open(self.diagnostics_path, "a", encoding="utf-8", buffering=1)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.compute_path_client = ActionClient(self, ComputePathToPose, self.compute_path_action)

        self.path_pub = self.create_publisher(Path, self.path_topic, 10)
        self.goal_pub = self.create_publisher(PoseStamped, self.final_goal_topic, 10)
        self.global_path_pub = self.create_publisher(Path, self.global_path_topic, 5)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 5)
        map_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.debug_map_pub = self.create_publisher(
            OccupancyGrid, "/spu_bert/debug_rolling_map", map_qos
        )
        self.plan_context_pub = self.create_publisher(String, self.plan_context_topic, map_qos)
        self.create_subscription(String, self.goal_completion_topic, self._on_goal_completion, map_qos)

        self.create_subscription(Odometry, self.odom_topic, self._on_odom, 20)
        self.create_subscription(Tracks, self.tracks_topic, self._on_tracks, 10)
        self.create_subscription(LaserScan, self.scan_topic, self._on_scan, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, self.goal_topic, self._on_goal, 10)
        self.create_timer(max(self.prediction_dt, 0.05), self._sample_histories)
        # Poll worker completion promptly; a 0.25s extra scheduling delay can
        # exhaust the tracker's 1.20s path lifetime on a roughly 1s inference.
        self.create_timer(0.05, self._on_timer)

        self._runtime = GuidedSpubertRuntime(
            repo_path=self.model_repo_path,
            config_path=self.model_config_path,
            checkpoint_path=self.model_checkpoint_path,
            map_provider=self.map_provider,
            use_cuda=self.use_cuda,
            d_sample=self.d_sample,
            runtime_seed=self.runtime_seed,
            guidance_radius=self.guidance_radius,
            tgp_top_k=self.tgp_top_k,
            footprint_radius=self.robot_radius + self.static_safety_margin,
            logger=self.get_logger(),
        )
        if self._runtime.obs_len != self.obs_len or self._runtime.pred_len != self.pred_len:
            raise RuntimeError(
                "runtime/model sequence mismatch: "
                f"node=({self.obs_len},{self.pred_len}) "
                f"model=({self._runtime.obs_len},{self._runtime.pred_len})"
            )
        # Only this worker touches the model or its private map snapshot. ROS
        # callbacks remain on the normal single-threaded executor.
        self._inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="spubert_inference")
        self._status("ready_disarmed_path_output_only")
        self.get_logger().info(
            "Real Jackal SPU-BERT bridge ready; this node publishes paths only. "
            "Use the separately armed safe_path_tracker for wheel commands."
        )

    def _string_param(self, name: str, default: str) -> str:
        return str(self.declare_parameter(name, default).value)

    def _bool_param(self, name: str, default: bool) -> bool:
        value = self.declare_parameter(name, default).value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_odom(self, msg: Odometry) -> None:
        frame = msg.header.frame_id or "odom"
        if self._last_pose_frame and not same_frame_id(frame, self._last_pose_frame):
            self._pose_frame_generation += 1
            self._robot_history.clear()
            self._robot_history_times.clear()
            self._robot_history_odom_stamps.clear()
            self._human_histories.clear()
            self._latest_humans = {}
            self._tracks_frame_valid = False
            self._previous_path = None
            self.heading_selector.reset()
        self._last_pose_frame = frame
        self._latest_odom = msg
        self._odom_stamp_s = self._now_s()

    def _on_scan(self, msg: LaserScan) -> None:
        self._latest_scan = msg
        self._scan_stamp_s = self._now_s()
        self._latest_scan_frame = str(msg.header.frame_id)
        self._scan_frame_valid = bool(self._latest_scan_frame) and same_frame_id(
            self._latest_scan_frame, self.base_frame
        )
        self._scan_data_error = ""
        if (not msg.ranges or not math.isfinite(msg.angle_min)
                or not math.isfinite(msg.angle_increment)
                or (len(msg.ranges) > 1 and msg.angle_increment == 0)
                or not any((math.isfinite(value) and value > 0) or value == math.inf for value in msg.ranges)):
            self._scan_data_error = "invalid_scan_geometry_or_ranges"

    def _on_tracks(self, msg: Tracks) -> None:
        self._tracks_stamp_s = self._now_s()
        self._tracks_source_stamp = getattr(msg.header, "stamp", None)
        self._latest_tracks_frame = str(msg.header.frame_id)
        self._tracks_data_error = ""
        robot = self._robot_pose()
        if robot is None:
            self._tracks_frame_valid = False
            return
        robot_x, robot_y, robot_yaw, odom_frame = robot
        latest: Dict[int, XY] = {}
        if self.tracks_are_robot_relative:
            self._tracks_frame_valid = bool(self._latest_tracks_frame) and same_frame_id(
                self._latest_tracks_frame, self.base_frame
            )
            if not self._tracks_frame_valid:
                self._latest_humans = {}
                return
        else:
            self._tracks_frame_valid = bool(self._latest_tracks_frame)
            if not self._tracks_frame_valid:
                self._latest_humans = {}
                return
        for track in msg.tracks:
            if not math.isfinite(float(track.score)):
                self._tracks_data_error = "nonfinite_pedestrian_track"
                self._latest_humans = {}
                return
            if float(track.score) < self.minimum_track_score:
                continue
            point = float(track.pose.position.x), float(track.pose.position.y)
            if not all(math.isfinite(value) for value in point):
                self._tracks_data_error = "nonfinite_pedestrian_track"
                self._latest_humans = {}
                return
            if self.tracks_are_robot_relative:
                world = transform_xy(point, (robot_x, robot_y), robot_yaw)
            else:
                world = self._xy_in_frame(point, self._latest_tracks_frame, odom_frame)
                if world is None:
                    self._tracks_frame_valid = False
                    self._latest_humans = {}
                    return
            latest[int(track.id)] = world
        self._latest_humans = latest

    def _on_goal(self, msg: PoseStamped) -> None:
        self._goal = msg
        self._goal_generation += 1
        self._mission.start()
        self._goal_xy = None
        self._goal_frame = ""
        self._previous_path = None
        self.heading_selector.reset()
        self._route_generation = -1
        self._route_message = None
        self._route_pending = False
        self._last_route_request_s = -math.inf
        self._hold("new_goal_waiting_for_route")
        self._status("new_goal_waiting_for_route")
        if self.require_global_path:
            self._request_route()

    def _on_goal_completion(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            goal_id = payload.get("goal_id", "")
        except (ValueError, TypeError, AttributeError):
            return
        if self._mission.complete(goal_id):
            self._record("goal_completed", {"goal_id": goal_id, "source": "tracker"})
            self._hold("goal_reached")

    def _publish_plan_context(self, state: str, reason: str = "", path_stamp_ns=None, path_points=None) -> None:
        if not self._mission.goal_id:
            return
        self.plan_context_pub.publish(String(data=json.dumps({
            "goal_id": self._mission.goal_id,
            "state": state,
            "frame_id": self._goal_frame,
            "goal": list(self._goal_xy) if self._goal_xy is not None else None,
            "path_stamp_ns": path_stamp_ns,
            "path": [list(point) for point in path_points] if path_points is not None else [],
            "reason": reason,
        }, allow_nan=False)))

    def _sample_histories(self) -> None:
        robot = self._robot_pose()
        now = self._now_s()
        odom_stamp_bad = self.check_sensor_header_stamps and self._latest_odom is not None and stamp_error(
            getattr(self._latest_odom.header, "stamp", None),
            now_ns=self.get_clock().now().nanoseconds, timeout_sec=self.odom_timeout,
            future_tolerance_sec=self.sensor_stamp_future_tolerance,
        )
        if robot is None or now - self._odom_stamp_s > self.odom_timeout or odom_stamp_bad:
            self._robot_history.clear()
            self._robot_history_times.clear()
            self._robot_history_odom_stamps.clear()
            self._human_histories.clear()
            self.heading_selector.reset()
            return
        if not all(math.isfinite(value) for value in robot[:3]):
            return
        self._robot_history.append((robot[0], robot[1]))
        self._robot_history_times.append(now)
        self._robot_history_odom_stamps.append(self._stamp_ns(self._latest_odom.header.stamp))
        if now - self._tracks_stamp_s > self.tracks_timeout:
            self._latest_humans = {}
        for track_id, point in self._latest_humans.items():
            history = self._human_histories.setdefault(track_id, deque(maxlen=self.obs_len))
            history.append(point)
        for track_id in set(self._human_histories) - set(self._latest_humans):
            self._human_histories.pop(track_id, None)

    def _robot_pose(self):
        self._pose_error = ""
        if self._latest_odom is None:
            return None
        pose = self._latest_odom.pose.pose
        q = pose.orientation
        if not all(math.isfinite(value) for value in (pose.position.x, pose.position.y, q.x, q.y, q.z, q.w)):
            self._pose_error = "nonfinite_robot_pose"
            return None
        if sum(value * value for value in (q.x, q.y, q.z, q.w)) <= 1e-12:
            self._pose_error = "invalid_robot_orientation"
            return None
        frame = self._latest_odom.header.frame_id or "odom"
        return (
            float(pose.position.x),
            float(pose.position.y),
            yaw_from_quaternion(q.x, q.y, q.z, q.w),
            frame,
        )

    def _on_timer(self) -> None:
        now = self._now_s()
        if self._pending_inference is not None:
            future, job = self._pending_inference
            if future.done():
                self._pending_inference = None
                self._finish_inference(future, job)
                # A completed job must not consume another poll interval before
                # the next eligible start. replan_period still limits starts.
                now = self._now_s()
            else:
                error = self._input_safety_error(now)
                if self._mission.completed:
                    self._hold("goal_reached")
                elif error:
                    self._hold(error)
                elif self._inference_age(job) > self.inference_max_age:
                    if not job.get("timeout_reported", False):
                        self._record("prediction", {**job["diagnostics"], "valid": False,
                                                    "reason": "inference_result_expired",
                                                    "inference_result_age_sec": self._inference_age(job),
                                                    "attempts": []})
                        job["timeout_reported"] = True
                    self._hold("inference_result_expired")
                return
        if now - self._last_predict_s < self.replan_period:
            return
        self._last_predict_s = now

        if self._mission.completed:
            self._hold("goal_reached")
            return

        robot = self._robot_pose()
        if robot is None or now - self._odom_stamp_s > self.odom_timeout:
            self._hold(self._pose_error or "odom_missing_or_stale")
            return
        if self._latest_scan is None or now - self._scan_stamp_s > self.scan_timeout:
            self._hold("scan_missing_or_stale")
            return
        if self._scan_data_error:
            self._hold(self._scan_data_error)
            return
        if not self._scan_frame_valid:
            self._hold(
                f"scan_frame_mismatch:got={self._latest_scan_frame or '<empty>'},"
                f"expected={self.base_frame}"
            )
            return
        if now - self._tracks_stamp_s > self.tracks_timeout:
            self._hold("pedestrian_tracker_heartbeat_stale")
            return
        if self._tracks_data_error:
            self._hold(self._tracks_data_error)
            return
        source_error = self._sensor_stamp_error()
        if source_error:
            self._hold(source_error)
            return
        if not self._tracks_frame_valid:
            self._hold(
                f"tracks_frame_mismatch:got={self._latest_tracks_frame or '<empty>'},"
                f"expected={self.base_frame}"
            )
            return
        if self._goal is None:
            self._hold("waiting_for_goal", publish_empty=False)
            return
        if len(self._robot_history) < self.obs_len:
            self._hold(f"warming_history_{len(self._robot_history)}_of_{self.obs_len}")
            return

        if (
            self.require_global_path
            and self.route_refresh_period > 0.0
            and not self._route_pending
            and now - self._last_route_request_s >= self.route_refresh_period
        ):
            self._request_route()

        robot_x, robot_y, robot_yaw, odom_frame = robot
        if not all(math.isfinite(value) for value in robot[:3]):
            self._hold("nonfinite_robot_pose")
            return
        goal_xy = self._pose_in_frame(self._goal, odom_frame)
        if goal_xy is None:
            self._hold("goal_transform_unavailable")
            return
        if not all(math.isfinite(value) for value in goal_xy):
            self._hold("nonfinite_goal")
            return
        self._goal_xy, self._goal_frame = goal_xy, odom_frame
        if math.hypot(goal_xy[0] - robot_x, goal_xy[1] - robot_y) <= self.goal_tolerance:
            self._mission.complete(self._mission.goal_id)
            self._record("goal_completed", {"goal_id": self._mission.goal_id, "source": "bridge"})
            self._hold("goal_reached")
            return

        map_update_stamp_ns = self._refresh_map(robot)
        if not self.map_provider.ready:
            self._hold("rolling_map_unavailable")
            return

        route = self._route_in_frame(odom_frame)
        if route is None:
            if self.require_global_path:
                if not self._route_pending and now - self._last_route_request_s > 1.0:
                    self._request_route()
                self._hold("waiting_for_nav2_global_path")
                return
            route = [(robot_x, robot_y), goal_xy]

        footprint_radius = self.robot_radius + self.static_safety_margin
        spacing = max(self.map_provider.resolution * 0.5, 0.02)

        def direct_path_safe(points: Sequence[XY]) -> bool:
            dense = sample_polyline(points, max_spacing=spacing)
            return self.map_provider.path_collision_cost(
                dense, radius=footprint_radius, weight=1.0
            ) <= 0.0

        try:
            if self.use_adaptive_guidance:
                guidance = adaptive_guidance_point_along_path(
                    route,
                    current=(robot_x, robot_y),
                    final_goal=goal_xy,
                    max_radius=self.guidance_radius,
                    min_radius=self.guidance_min_distance,
                    probe_step=self.guidance_probe_step,
                    is_direct_path_safe=direct_path_safe,
                )
            else:
                from .guided_spubert_runtime import guidance_point_along_path

                guidance = guidance_point_along_path(
                    route,
                    current=(robot_x, robot_y),
                    final_goal=goal_xy,
                    radius=self.guidance_radius,
                )
        except ValueError as exc:
            self._hold(f"adaptive_guidance_failed:{exc}")
            return

        try:
            heading = self.heading_selector.select(
                history_xy=list(self._robot_history),
                sample_times_s=list(self._robot_history_times),
                odom_yaw_rad=robot_yaw,
            )
            inference_started_ns = int(self.get_clock().now().nanoseconds)
            runtime_inputs = dict(
                robot_history=list(self._robot_history),
                robot_yaw=robot_yaw,
                human_histories={key: list(value) for key, value in self._human_histories.items()},
                final_goal=goal_xy,
                guidance_point_world=guidance,
                heading_override_rad=heading.theta_rad,
                heading_source=heading.source,
            )
        except Exception as exc:  # keep the outdoor control process fail-closed
            self.get_logger().error(f"SPU-BERT inference failed: {exc}")
            self._record("inference_error", {"error": str(exc)})
            self._hold("inference_error")
            return

        input_diagnostics = {
            "goal_id": self._mission.goal_id,
            "model_output_collected": False,
            "robot": [robot_x, robot_y],
            "robot_yaw_rad": robot_yaw,
            "goal": list(goal_xy),
            "guidance": list(guidance),
            "heading": heading.diagnostics,
            "history_sample_times_s": list(self._robot_history_times),
            "history_odom_stamps_ns": list(self._robot_history_odom_stamps),
            "odom_stamp_ns": self._stamp_ns(self._latest_odom.header.stamp),
            "scan_stamp_ns": self._stamp_ns(self._latest_scan.header.stamp),
            "tracks_stamp_ns": self._stamp_ns(self._tracks_source_stamp) if self._tracks_source_stamp is not None else None,
            "odom_frame": odom_frame,
            "map_update_stamp_ns": map_update_stamp_ns,
            "model_map_generation": self._map_generation,
            "map_context": {"resolution": self.map_provider.resolution,
                            "origin": [self.map_provider.origin_x, self.map_provider.origin_y],
                            "width": self.map_provider.width, "height": self.map_provider.height},
            "odom_receipt_age_sec": self._now_s() - self._odom_stamp_s,
            "scan_receipt_age_sec": self._now_s() - self._scan_stamp_s,
            "tracks_receipt_age_sec": self._now_s() - self._tracks_stamp_s,
        }
        job = {
            "goal_id": self._mission.goal_id, "frame": odom_frame,
            "pose_frame_generation": self._pose_frame_generation,
            "start_s": inference_started_ns * 1e-9, "start_monotonic": time.monotonic(),
            "route": route, "guidance": guidance, "diagnostics": input_diagnostics,
        }
        dispatch_error = self._input_safety_error()
        if dispatch_error:
            self._record("prediction", {**input_diagnostics, "valid": False,
                                         "reason": "inputs_stale_before_inference",
                                         "current_input_error": dispatch_error, "attempts": []})
            self._hold("inputs_stale_before_inference")
            return
        # Snapshot contains its own numpy grid. The main thread can rebuild the
        # live map while inference reads this snapshot without a data race.
        snapshot = deepcopy(self.map_provider)
        future = self._inference_executor.submit(self._run_inference, snapshot, runtime_inputs)
        self._pending_inference = (future, job)

    def _run_inference(self, map_snapshot, runtime_inputs):
        started = time.monotonic()
        self._runtime.map_provider = map_snapshot
        candidates = self._runtime.predict_candidates(**runtime_inputs)
        return candidates, deepcopy(self._runtime.last_input_diagnostics), time.monotonic() - started

    def _inference_age(self, job) -> float:
        # A ROS clock reset must not make old work become young again.
        ros_age = self._now_s() - job["start_s"]
        if ros_age < 0:
            return math.inf
        return max(ros_age, time.monotonic() - job["start_monotonic"])

    def _input_safety_error(self, now=None) -> str:
        now = self._now_s() if now is None else now
        robot = self._robot_pose()
        if robot is None or now - self._odom_stamp_s > self.odom_timeout:
            return self._pose_error or "odom_missing_or_stale"
        if self._latest_scan is None or now - self._scan_stamp_s > self.scan_timeout:
            return "scan_missing_or_stale"
        if self._scan_data_error:
            return self._scan_data_error
        if not self._scan_frame_valid:
            return "scan_frame_mismatch"
        if now - self._tracks_stamp_s > self.tracks_timeout:
            return "pedestrian_tracker_heartbeat_stale"
        if self._tracks_data_error:
            return self._tracks_data_error
        if not self._tracks_frame_valid:
            return "tracks_frame_mismatch"
        return self._sensor_stamp_error()

    def _refresh_map(self, robot):
        robot_x, robot_y, robot_yaw, odom_frame = robot
        self.map_provider.update_from_scan(
            robot_x=robot_x, robot_y=robot_y, robot_yaw=robot_yaw,
            ranges=self._latest_scan.ranges, angle_min=self._latest_scan.angle_min,
            angle_increment=self._latest_scan.angle_increment,
            range_min=self._latest_scan.range_min, range_max=self._latest_scan.range_max,
            free_ray_limit_m=self.free_ray_limit,
        )
        self._map_generation += 1
        self._publish_debug_map(odom_frame)
        return int(self.get_clock().now().nanoseconds)

    def _current_human_histories(self):
        # Anchor the sampled velocity at the latest measured person position.
        # Translating all samples preserves velocity instead of treating an
        # irregular callback interval as one full prediction_dt. New IDs are
        # included immediately, even before the history timer next fires.
        current = {}
        for human_id, point in self._latest_humans.items():
            history = list(self._human_histories.get(human_id, []))
            if history:
                dx, dy = point[0] - history[-1][0], point[1] - history[-1][1]
                current[human_id] = [(x + dx, y + dy) for x, y in history]
            else:
                current[human_id] = [point]
        return current

    def _finish_inference(self, future, job) -> None:
        input_diagnostics = job["diagnostics"]
        age = self._inference_age(job)
        input_diagnostics["inference_result_age_sec"] = age

        def reject(reason):
            self._record("prediction", {**input_diagnostics, "valid": False, "reason": reason, "attempts": []})
            self._hold(reason)

        # Do not let an old goal or an old localization frame execute a result.
        if job["goal_id"] != self._mission.goal_id:
            reject("inference_goal_changed")
            return
        if self._mission.completed:
            reject("goal_reached")
            return
        if age > self.inference_max_age:
            reject("inference_result_expired")
            return
        error = self._input_safety_error()
        if error:
            input_diagnostics["current_input_error"] = error
            reject("inputs_stale_after_inference" if "stale" in error else error)
            return
        robot = self._robot_pose()
        robot_x, robot_y, robot_yaw, odom_frame = robot
        if (job["pose_frame_generation"] != self._pose_frame_generation
                or not same_frame_id(job["frame"], odom_frame)):
            reject("inference_pose_frame_changed")
            return
        goal_xy = self._pose_in_frame(self._goal, odom_frame)
        if goal_xy is None or not all(math.isfinite(value) for value in goal_xy):
            reject("goal_transform_unavailable")
            return
        self._goal_xy, self._goal_frame = goal_xy, odom_frame
        if math.hypot(goal_xy[0] - robot_x, goal_xy[1] - robot_y) <= self.goal_tolerance:
            self._mission.complete(self._mission.goal_id)
            reject("goal_reached")
            return
        try:
            candidates, model_diagnostics, duration = future.result()
        except Exception as exc:
            self.get_logger().error(f"SPU-BERT inference failed: {exc}")
            self._record("inference_error", {"error": str(exc), **input_diagnostics})
            self._hold("inference_error")
            return
        input_diagnostics["model_input"] = model_diagnostics
        input_diagnostics["model_output_collected"] = True
        input_diagnostics["inference_duration_sec"] = duration
        validation_map_stamp_ns = self._refresh_map(robot)
        if not self.map_provider.ready:
            reject("rolling_map_unavailable")
            return
        human_histories = self._current_human_histories()
        input_diagnostics["validation"] = {
            "robot": [robot_x, robot_y], "robot_yaw_rad": robot_yaw,
            "goal": list(goal_xy),
            "map_generation": self._map_generation,
            "map_update_stamp_ns": validation_map_stamp_ns,
            "odom_stamp_ns": self._stamp_ns(self._latest_odom.header.stamp),
            "scan_stamp_ns": self._stamp_ns(self._latest_scan.header.stamp),
            "tracks_stamp_ns": self._stamp_ns(self._tracks_source_stamp) if self._tracks_source_stamp else None,
            "history_sample_times_s": list(self._robot_history_times),
            "human_positions": {str(key): list(value[-1]) for key, value in human_histories.items()},
        }
        route, guidance = job["route"], job["guidance"]
        footprint_radius = self.robot_radius + self.static_safety_margin
        now = self._now_s()

        attempts = []
        selected: Optional[GuidedInferenceResult] = None
        selected_check = None
        options, checks = [], []
        for index, result in enumerate(candidates):
            if not result.selected_goal_valid:
                reason = "no_map_safe_goal"
                check = None
            elif not result.trajectory_map_safe or not result.execution_valid:
                reason = "model_map_check_failed"
                check = None
            elif len(result.path_world) != self.pred_len:
                reason = "invalid_path_length"
                check = None
            else:
                check = validate_candidate_path(
                    current=(robot_x, robot_y),
                    path=result.path_world,
                    final_goal=goal_xy,
                    map_provider=self.map_provider,
                    footprint_radius=footprint_radius,
                    prediction_dt=self.prediction_dt,
                    maximum_model_speed=self.maximum_model_speed,
                    maximum_step_ratio=self.maximum_step_ratio,
                    minimum_goal_progress=self.minimum_goal_progress,
                    human_histories=human_histories,
                    human_sample_dt=self.prediction_dt,
                    minimum_human_center_distance=self.minimum_human_center_distance,
                    human_radius=self.human_radius,
                    human_safety_margin=self.human_safety_margin,
                )
                reason = check.reason
            valid = check.valid if check is not None else False
            if valid and (not math.isfinite(result.guidance_distance_m) or result.guidance_distance_m < 0):
                valid, reason = False, "invalid_guidance_distance"
            first_collision = None
            if check is not None and check.footprint_collision_count > 0:
                first_collision = self.map_provider.first_path_collision(
                    sample_polyline([(robot_x, robot_y), *result.path_world],
                                    max_spacing=max(self.map_provider.resolution * 0.5, 0.02)),
                    radius=footprint_radius,
                )
            attempts.append(
                {
                    "rank": int(result.candidate_rank),
                    "candidate_index": int(result.candidate_index),
                    "valid": bool(valid),
                    "reason": reason,
                    "path": [list(point) for point in result.path_world],
                    "footprint_collision_count": None if check is None else check.footprint_collision_count,
                    "first_collision": first_collision,
                    "maximum_step_m": None if check is None or not math.isfinite(check.maximum_step_m) else check.maximum_step_m,
                    "goal_progress_m": None if check is None else check.goal_progress_m,
                    "minimum_human_distance_m": (
                        None if check is None or not math.isfinite(check.minimum_human_distance_m)
                        else check.minimum_human_distance_m
                    ),
                }
            )
            checks.append(check)
            options.append(PathOption(index, result.path_world, float(result.guidance_distance_m), bool(valid)))

        reference = self._previous_path if (
            self._previous_path is not None
            and same_frame_id(self._previous_path_frame, odom_frame)
            and 0 <= now - self._previous_path_stamp_s <= self.continuity_reference_timeout
        ) else None
        ranked = rank_valid_paths(options, current=(robot_x, robot_y), robot_yaw=robot_yaw,
                                  previous_path=reference, config=self.selection_config)
        for score in ranked:
            attempts[score.index]["selection_score"] = asdict(score)
        if ranked:
            selected = candidates[ranked[0].index]
            selected_check = checks[ranked[0].index]
        input_diagnostics["selection_mode"] = self.selection_config.mode
        input_diagnostics["continuity_reference_used"] = reference is not None

        debug_result = selected or (candidates[0] if candidates else None)
        self._publish_markers(
            odom_frame,
            route,
            guidance,
            debug_result,
            selected is not None,
        )
        if selected is None:
            reason = attempts[0]["reason"] if attempts else "no_candidates"
            self._record("prediction", {**input_diagnostics, "valid": False, "reason": reason, "attempts": attempts})
            self._hold(f"all_candidates_rejected:{reason}")
            return

        publish_now = self._now_s()
        source_error = self._sensor_stamp_error()
        if self._inference_age(job) > self.inference_max_age:
            self._record("prediction", {**input_diagnostics, "valid": False,
                                         "reason": "inference_result_expired_after_validation",
                                         "inference_result_age_sec": self._inference_age(job), "attempts": attempts})
            self._hold("inference_result_expired_after_validation")
            return
        if (source_error or publish_now - self._odom_stamp_s > self.odom_timeout
                or publish_now - self._scan_stamp_s > self.scan_timeout
                or publish_now - self._tracks_stamp_s > self.tracks_timeout):
            self._record("prediction", {**input_diagnostics, "valid": False, "reason": "inputs_stale_after_validation", "source_stamp_error": source_error, "attempts": attempts})
            self._hold("inputs_stale_after_validation")
            return
        path_message = self._path_message(odom_frame, selected.path_world)
        path_stamp_ns = self._stamp_ns(path_message.header.stamp)
        self._publish_plan_context("active", path_stamp_ns=path_stamp_ns, path_points=selected.path_world)
        self.path_pub.publish(path_message)
        self.goal_pub.publish(self._point_pose(odom_frame, goal_xy))
        self._previous_path = [(robot_x, robot_y), *selected.path_world]
        self._previous_path_stamp_s = self._now_s()
        self._previous_path_frame = odom_frame
        self._status(
            f"path_valid rank={selected.candidate_rank}/{len(candidates)} "
            f"human_min={selected_check.minimum_human_distance_m:.3f}"
        )
        self._record(
            "prediction",
            {
                **input_diagnostics,
                "valid": True,
                "robot": [robot_x, robot_y],
                "goal": list(goal_xy),
                "guidance": list(guidance),
                "selected_rank": int(selected.candidate_rank),
                "selected_candidate_index": int(selected.candidate_index),
                "path": [list(point) for point in selected.path_world],
                "attempts": attempts,
                "path_stamp_ns": path_stamp_ns,
            },
        )

    def _request_route(self) -> bool:
        if self._goal is None or self._route_pending or self._mission.completed:
            return False
        self._last_route_request_s = self._now_s()
        if not self.compute_path_client.wait_for_server(timeout_sec=0.05):
            self._status("compute_path_to_pose_unavailable")
            return False
        generation = self._goal_generation
        request = ComputePathToPose.Goal()
        request.goal = self._goal
        request.use_start = False
        request.planner_id = self.planner_id
        self._route_pending = True
        future = self.compute_path_client.send_goal_async(request)
        future.add_done_callback(
            lambda done, generation=generation: self._on_route_goal_response(done, generation)
        )
        self._status("requesting_nav2_global_path")
        return True

    def _on_route_goal_response(self, future, generation: int) -> None:
        if generation != self._goal_generation or self._mission.completed:
            return
        try:
            handle = future.result()
        except Exception as exc:
            self._route_pending = False
            self._status(f"compute_path_request_failed:{exc}")
            return
        if handle is None or not handle.accepted:
            self._route_pending = False
            self._status("compute_path_rejected")
            return
        handle.get_result_async().add_done_callback(
            lambda done, generation=generation: self._on_route_result(done, generation)
        )

    def _on_route_result(self, future, generation: int) -> None:
        if generation != self._goal_generation or self._mission.completed:
            return
        self._route_pending = False
        try:
            message = future.result().result.path
        except Exception as exc:
            self._status(f"compute_path_result_failed:{exc}")
            return
        if len(message.poses) < 2:
            self._status("compute_path_empty")
            return
        self._route_message = message
        self._route_generation = generation
        self._status(f"nav2_global_path_ready poses={len(message.poses)}")

    def _route_in_frame(self, target_frame: str):
        if self._route_message is None or self._route_generation != self._goal_generation:
            return None
        source_frame = self._route_message.header.frame_id or target_frame
        points = []
        for pose in self._route_message.poses:
            point = self._xy_in_frame(
                (pose.pose.position.x, pose.pose.position.y), source_frame, target_frame
            )
            if point is None:
                return None
            points.append(point)
        transformed = self._path_message(target_frame, points)
        self.global_path_pub.publish(transformed)
        return points

    def _pose_in_frame(self, pose: PoseStamped, target_frame: str):
        return self._xy_in_frame(
            (pose.pose.position.x, pose.pose.position.y),
            pose.header.frame_id or target_frame,
            target_frame,
        )

    def _xy_in_frame(self, point: XY, source_frame: str, target_frame: str):
        if not source_frame or source_frame == target_frame:
            return float(point[0]), float(point[1])
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=0.10),
            )
        except TransformException as exc:
            self.get_logger().warning(
                f"TF unavailable {source_frame}->{target_frame}: {exc}",
                throttle_duration_sec=2.0,
            )
            return None
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        yaw = yaw_from_quaternion(rotation.x, rotation.y, rotation.z, rotation.w)
        return transform_xy(point, (translation.x, translation.y), yaw)

    def _publish_debug_map(self, frame_id: str) -> None:
        message = OccupancyGrid()
        message.header.frame_id = frame_id
        message.header.stamp = self.get_clock().now().to_msg()
        message.info.resolution = float(self.map_provider.resolution)
        message.info.width = int(self.map_provider.width)
        message.info.height = int(self.map_provider.height)
        message.info.origin.position.x = float(self.map_provider.origin_x)
        message.info.origin.position.y = float(self.map_provider.origin_y)
        message.info.origin.orientation.w = 1.0
        message.data = self.map_provider.occupancy_grid_data()
        self.debug_map_pub.publish(message)

    def _publish_markers(
        self,
        frame_id: str,
        route: Sequence[XY],
        guidance: XY,
        result: Optional[GuidedInferenceResult],
        valid: bool,
    ) -> None:
        array = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        array.markers.append(clear)
        array.markers.append(
            self._line_marker(frame_id, "global_route", 1, route, (0.05, 0.60, 0.35, 0.75), 0.06)
        )
        guidance_marker = Marker()
        guidance_marker.header.frame_id = frame_id
        guidance_marker.header.stamp = self.get_clock().now().to_msg()
        guidance_marker.ns = "adaptive_guidance"
        guidance_marker.id = 2
        guidance_marker.type = Marker.SPHERE
        guidance_marker.action = Marker.ADD
        guidance_marker.pose.position.x = float(guidance[0])
        guidance_marker.pose.position.y = float(guidance[1])
        guidance_marker.pose.position.z = 0.18
        guidance_marker.pose.orientation.w = 1.0
        guidance_marker.scale.x = guidance_marker.scale.y = guidance_marker.scale.z = 0.35
        guidance_marker.color = ColorRGBA(r=1.0, g=0.72, b=0.0, a=1.0)
        array.markers.append(guidance_marker)
        if result is not None:
            for index, point in enumerate(result.candidate_goals_world):
                marker = Marker()
                marker.header.frame_id = frame_id
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "mgp_candidates"
                marker.id = 100 + index
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = float(point[0])
                marker.pose.position.y = float(point[1])
                marker.pose.position.z = 0.10
                marker.pose.orientation.w = 1.0
                marker.scale.x = marker.scale.y = marker.scale.z = 0.16
                safe = index < len(result.candidate_safe_mask) and result.candidate_safe_mask[index]
                marker.color = ColorRGBA(
                    r=0.10 if safe else 0.90,
                    g=0.75 if safe else 0.15,
                    b=0.35 if safe else 0.10,
                    a=0.90,
                )
                array.markers.append(marker)
            array.markers.append(
                self._line_marker(
                    frame_id,
                    "selected_tgp",
                    3,
                    result.path_world,
                    (0.10, 0.35, 0.95, 1.0) if valid else (0.90, 0.10, 0.10, 1.0),
                    0.10,
                )
            )
        self.marker_pub.publish(array)

    def _line_marker(self, frame_id, namespace, marker_id, points, color, width):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = float(width)
        marker.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])
        marker.points = [Point(x=float(x), y=float(y), z=0.08) for x, y in points]
        return marker

    def _path_message(self, frame_id: str, points: Sequence[XY]) -> Path:
        message = Path()
        message.header.frame_id = frame_id
        now = self.get_clock().now()
        message.header.stamp = now.to_msg()
        for index, point in enumerate(points):
            pose = PoseStamped()
            pose.header.frame_id = frame_id
            pose.header.stamp = (
                now + Duration(seconds=(index + 1) * self.prediction_dt)
            ).to_msg()
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            pose.pose.orientation.w = 1.0
            message.poses.append(pose)
        return message

    def _point_pose(self, frame_id: str, point: XY) -> PoseStamped:
        message = PoseStamped()
        message.header.frame_id = frame_id
        message.header.stamp = self.get_clock().now().to_msg()
        message.pose.position.x = float(point[0])
        message.pose.position.y = float(point[1])
        message.pose.orientation.w = 1.0
        return message

    def _hold(self, reason: str, publish_empty: bool = True) -> None:
        self._previous_path = None
        self._publish_plan_context("complete" if self._mission.completed else "hold", reason)
        self._status(f"hold:{reason}")
        if not publish_empty:
            return
        frame = self._latest_odom.header.frame_id if self._latest_odom is not None else "odom"
        self.path_pub.publish(self._path_message(frame or "odom", []))

    @staticmethod
    def _stamp_ns(stamp) -> int:
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _sensor_stamp_error(self) -> str:
        if not self.check_sensor_header_stamps:
            return ""
        now_ns = self.get_clock().now().nanoseconds
        sources = (
            ("odom", getattr(getattr(self._latest_odom, "header", None), "stamp", None), self.odom_timeout),
            ("scan", getattr(getattr(self._latest_scan, "header", None), "stamp", None), self.scan_timeout),
            ("tracks", self._tracks_source_stamp, self.tracks_timeout),
        )
        for name, stamp, timeout in sources:
            error = stamp_error(stamp, now_ns=now_ns, timeout_sec=timeout,
                                future_tolerance_sec=self.sensor_stamp_future_tolerance)
            if error:
                return f"{name}_stamp_{error}"
        return ""

    def _status(self, text: str) -> None:
        if text == self._last_status:
            return
        self._last_status = text
        self.status_pub.publish(String(data=text))
        self.get_logger().info(text)

    def _record(self, event: str, payload: dict) -> None:
        if self._diagnostics_file is None:
            return
        record = {"event": event, "stamp_ns": int(self.get_clock().now().nanoseconds), **payload}
        try:
            self._diagnostics_file.write(json.dumps(json_finite(record), ensure_ascii=False, allow_nan=False) + "\n")
        except OSError as exc:
            failed_file, self._diagnostics_file = self._diagnostics_file, None
            try:
                failed_file.close()
            except OSError:
                pass
            self.get_logger().warning(f"Bridge diagnostics disabled after a write failure: {exc}")

    def destroy_node(self):
        self._inference_executor.shutdown(wait=True, cancel_futures=True)
        if self._diagnostics_file is not None:
            self._diagnostics_file.close()
            self._diagnostics_file = None
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RealJackalSpubertBridgeNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
