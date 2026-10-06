#!/usr/bin/env python3
from __future__ import annotations

import math
import os
from collections import deque
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
    lateral_guidance_point,
    sample_polyline,
)
from .rolling_laser_map import RollingLaserMapProvider
from .arc_planner import arc_cost, arc_points, curvature_fan
from .candidate_select import SelectionConfig, choose_candidate
from .route_fallback import lateral_offsets, route_prefix_points, shift_path_laterally
from .collision_diagnostics import collision_evidence, header_evidence
from .tracker_diagnostics import diagnostic_json


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
        self.collision_marker_topic = self._string_param(
            "collision_marker_topic", "/spu_bert/collision_markers"
        )
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

        # If the guidance point / all candidates keep getting rejected as
        # blocked for this long, stop holding in place and sweep-scan instead
        # (09/21 log: continuous holds of ~167s and ~105s at the same spot
        # with zero recovery in between).
        self.stuck_hold_timeout_sec = float(
            self.declare_parameter("stuck_hold_timeout_sec", 3.0).value
        )
        self.recovery_sweep_angle_rad = float(
            self.declare_parameter("recovery_sweep_angle_rad", 1.10).value
        )
        self.recovery_sweep_period_sec = float(
            self.declare_parameter("recovery_sweep_period_sec", 2.5).value
        )
        self.recovery_step_m = float(
            self.declare_parameter("recovery_step_m", 1.00).value
        )
        # Turning in place sweeps a circle of the body's circumscribed radius
        # (Jackal 0.51 x 0.43 m -> ~0.33 m). Do not start a recovery rotation
        # with anything inside that circle plus this margin: it would swing a
        # corner into the obstacle (2026-10-06: robot stalled beside a box).
        # 0 disables the guard.
        self.recovery_rotation_clearance_m = float(
            self.declare_parameter("recovery_rotation_clearance_m", 0.38).value
        )
        # When every SPU-BERT candidate is rejected, follow a prefix of the
        # Nav2 global path instead of stopping -- but only if that prefix
        # passes the very same validate_candidate_path checks as a model
        # candidate (footprint vs. live rolling map, pedestrian distance,
        # goal progress). The speed limit stays the tracker's own. The
        # fallback is time-boxed per episode so it cannot mask a persistent
        # problem; after that the normal blocked-hold/recovery takes over.
        self.global_path_fallback = bool(
            self.declare_parameter("global_path_fallback", True).value
        )
        self.global_fallback_length_m = float(
            self.declare_parameter("global_fallback_length_m", 1.5).value
        )
        self.global_fallback_min_safe_length_m = float(
            self.declare_parameter("global_fallback_min_safe_length_m", 0.8).value
        )
        self.global_fallback_max_sec = float(
            self.declare_parameter("global_fallback_max_sec", 20.0).value
        )
        # The Nav2 route only knows the static map, so around a live obstacle
        # in a narrow corridor the safe tube can sit a few centimetres off the
        # route. Try the route prefix slid sideways in small steps (smallest
        # shift that keeps the longest safe prefix wins).
        self.global_fallback_max_shift_m = float(
            self.declare_parameter("global_fallback_max_shift_m", 0.30).value
        )
        self.global_fallback_shift_step_m = float(
            self.declare_parameter("global_fallback_shift_step_m", 0.03).value
        )
        self.global_fallback_taper_m = float(
            self.declare_parameter("global_fallback_taper_m", 0.40).value
        )
        # Model-free last resort: constant-curvature arcs from the current pose,
        # each validated exactly like a model candidate; the one ending nearest
        # the route point ahead wins. Handles obstacles the sideways-shifted
        # route cannot clear.
        self.arc_planner_fallback = bool(
            self.declare_parameter("arc_planner_fallback", True).value
        )
        self.arc_max_curvature = float(self.declare_parameter("arc_max_curvature", 1.6).value)
        self.arc_curvature_step = float(self.declare_parameter("arc_curvature_step", 0.1).value)
        self.arc_target_distance_m = float(
            self.declare_parameter("arc_target_distance_m", 2.0).value
        )
        # Start directions tried relative to the robot heading (rad). A large
        # offset makes the tracker pivot in place first, which is how the robot
        # gets around something directly in front of it.
        # Default offsets stay below the tracker's rotate-in-place angle (0.95 rad)
        # so avoidance is a curve driven at speed, not stop-pivot-go. 1.2 rad would
        # make the tracker zero the forward speed and turn on the spot.
        self.arc_heading_offsets = tuple(
            float(v) for v in self.declare_parameter(
                "arc_heading_offsets_rad", [0.0, 0.4, -0.4, 0.8, -0.8]
            ).value
        )
        # Reduced footprint radius for a last "keep moving" tier when nothing fits
        # at the normal radius (robot centre must still be clear at this radius).
        self.escape_footprint_radius_m = float(
            self.declare_parameter("escape_footprint_radius_m", 0.36).value
        )
        # The escape radius adapts to the robot's actual clearance: the largest radius
        # (<= the value above) at which the robot's own position is free, but never
        # below this floor (body circumscribed radius ~0.33 m). Paths must then stay at
        # least that far from every obstacle, i.e. never closer than the robot is now.
        self.escape_floor_radius_m = float(
            self.declare_parameter("escape_floor_radius_m", 0.33).value
        )
        # Choosing among valid model candidates. All weights 0 keeps the old rule
        # (first valid by model rank). With weights, a candidate that jumps sideways
        # relative to the path published a moment ago, or leaves the robot heading
        # sharply, is avoided: the tracker otherwise chases a target that flips between
        # left and right and ends up stopping to turn (2026-10-06 field run).
        self.selection_config = SelectionConfig(
            rank_weight=float(self.declare_parameter("selection_rank_weight", 0.0).value),
            continuity_weight=float(self.declare_parameter("selection_continuity_weight", 0.0).value),
            heading_weight=float(self.declare_parameter("selection_heading_weight", 0.0).value),
            heading_limit_rad=float(self.declare_parameter("selection_heading_limit_rad", 0.0).value),
        )

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
        self._latest_scan_frame = ""
        self._latest_tracks_frame = ""
        self._latest_humans: Dict[int, XY] = {}
        self._robot_history = deque(maxlen=self.obs_len)
        self._human_histories: Dict[int, deque[XY]] = {}
        self._odom_stamp_s = -math.inf
        self._scan_stamp_s = -math.inf
        self._tracks_stamp_s = -math.inf
        self._goal: Optional[PoseStamped] = None
        self._goal_generation = 0
        self._route_generation = -1
        self._route_message: Optional[Path] = None
        self._route_pending = False
        self._last_route_request_s = -math.inf
        self._last_predict_s = -math.inf
        self._last_status = ""
        self._diagnostics_file = None
        self._blocked_since_s: Optional[float] = None
        self._recovery_active = False
        self._global_fallback_since_s: Optional[float] = None
        self._last_published_path = None
        self._last_published_s = -math.inf
        self._global_fallback_last_s = -math.inf
        self._throttled_record_s: Dict[str, float] = {}
        self._recovery_started_s = -math.inf

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
        self.collision_marker_pub = self.create_publisher(MarkerArray, self.collision_marker_topic, 5)
        map_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.debug_map_pub = self.create_publisher(
            OccupancyGrid, "/spu_bert/debug_rolling_map", map_qos
        )

        self.create_subscription(Odometry, self.odom_topic, self._on_odom, 20)
        self.create_subscription(Tracks, self.tracks_topic, self._on_tracks, 10)
        self.create_subscription(LaserScan, self.scan_topic, self._on_scan, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, self.goal_topic, self._on_goal, 10)
        self.create_timer(max(self.prediction_dt, 0.05), self._sample_histories)
        self.create_timer(max(min(self.replan_period, 0.25), 0.05), self._on_timer)

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
        self._latest_odom = msg
        self._odom_stamp_s = self._now_s()

    def _on_scan(self, msg: LaserScan) -> None:
        self._latest_scan = msg
        self._scan_stamp_s = self._now_s()
        self._latest_scan_frame = str(msg.header.frame_id)
        self._scan_frame_valid = bool(self._latest_scan_frame) and same_frame_id(
            self._latest_scan_frame, self.base_frame
        )

    def _on_tracks(self, msg: Tracks) -> None:
        self._tracks_stamp_s = self._now_s()
        self._latest_tracks_frame = str(msg.header.frame_id)
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
            if float(track.score) < self.minimum_track_score:
                continue
            point = float(track.pose.position.x), float(track.pose.position.y)
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
        self._route_generation = -1
        self._route_message = None
        self._route_pending = False
        self._last_route_request_s = -math.inf
        self._blocked_since_s = None
        self._recovery_active = False
        self._status("new_goal_waiting_for_route")
        if self.require_global_path:
            self._request_route()

    def _sample_histories(self) -> None:
        robot = self._robot_pose()
        if robot is None:
            return
        self._robot_history.append((robot[0], robot[1]))
        now = self._now_s()
        if now - self._tracks_stamp_s > self.tracks_timeout:
            self._latest_humans = {}
        for track_id, point in self._latest_humans.items():
            history = self._human_histories.setdefault(track_id, deque(maxlen=self.obs_len))
            history.append(point)
        for track_id in set(self._human_histories) - set(self._latest_humans):
            self._human_histories.pop(track_id, None)

    def _robot_pose(self):
        if self._latest_odom is None:
            return None
        pose = self._latest_odom.pose.pose
        q = pose.orientation
        frame = self._latest_odom.header.frame_id or "odom"
        return (
            float(pose.position.x),
            float(pose.position.y),
            yaw_from_quaternion(q.x, q.y, q.z, q.w),
            frame,
        )

    def _on_timer(self) -> None:
        now = self._now_s()
        if now - self._last_predict_s < self.replan_period:
            return
        self._last_predict_s = now

        robot = self._robot_pose()
        if robot is None or now - self._odom_stamp_s > self.odom_timeout:
            self._hold("odom_missing_or_stale")
            return
        if self._latest_scan is None or now - self._scan_stamp_s > self.scan_timeout:
            self._hold("scan_missing_or_stale")
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
        goal_xy = self._pose_in_frame(self._goal, odom_frame)
        if goal_xy is None:
            self._hold("goal_transform_unavailable")
            return
        if math.hypot(goal_xy[0] - robot_x, goal_xy[1] - robot_y) <= self.goal_tolerance:
            self._hold("goal_reached")
            return

        # Snapshot the exact latest-pose/latest-scan combination used below.
        # This records existing timing behavior; it does not time-align inputs.
        map_update_s = self._now_s()
        collision_context = {
            "schema_version": 1,
            "map_update_stamp_ns": int(map_update_s * 1e9),
            "frame_id": odom_frame,
            "robot_pose": [robot_x, robot_y, robot_yaw],
            "scan": header_evidence(self._latest_scan, self._scan_stamp_s, map_update_s),
            "odom": header_evidence(self._latest_odom, self._odom_stamp_s, map_update_s),
            "pose_policy": "latest_odom_at_map_update_not_scan_time_aligned",
            "robot_radius_m": self.robot_radius,
            "static_safety_margin_m": self.static_safety_margin,
            "footprint_radius_m": self.robot_radius + self.static_safety_margin,
            "map_resolution_m": self.map_provider.resolution,
            "unknown_is_occupied": self.map_provider.unknown_is_occupied,
        }
        self.map_provider.update_from_scan(
            robot_x=robot_x,
            robot_y=robot_y,
            robot_yaw=robot_yaw,
            ranges=self._latest_scan.ranges,
            angle_min=self._latest_scan.angle_min,
            angle_increment=self._latest_scan.angle_increment,
            range_min=self._latest_scan.range_min,
            range_max=self._latest_scan.range_max,
            free_ray_limit_m=self.free_ray_limit,
        )
        collision_context["map_origin_xy"] = [self.map_provider.origin_x, self.map_provider.origin_y]
        collision_context["map_shape_cells"] = [self.map_provider.width, self.map_provider.height]
        self._publish_debug_map(odom_frame)
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
            # The along-path search never leaves the Nav2 route, so a live
            # obstacle (not on the static map) that straddles the whole
            # route within the footprint radius fails it everywhere. Try a
            # lateral fan around the robot's own heading before escalating
            # to _blocked_hold; it's checked with the same live rolling-map
            # safety function, and whatever SPU-BERT actually predicts from
            # it is still independently validated below like any other
            # candidate (field-tested 2026-09-29: resolved 3 consecutive
            # otherwise-stuck cycles).
            detour = lateral_guidance_point((robot_x, robot_y), robot_yaw, direct_path_safe)
            if detour is None:
                self._record_throttled("lateral_detour_failed", {
                    "reason": "no_lateral_point_passed_footprint_check",
                    "robot": [robot_x, robot_y, robot_yaw],
                    "guidance_error": str(exc),
                    "clearance_probe": self._clearance_probe(robot_x, robot_y),
                }, now)
                if self._try_global_path_fallback(
                    f"adaptive_guidance_failed:{exc}", route, goal_xy, robot_x, robot_y,
                    robot_yaw, odom_frame, now,
                ):
                    return
                self._blocked_hold(
                    f"adaptive_guidance_failed:{exc}", now, robot_x, robot_y, robot_yaw, odom_frame
                )
                return
            guidance = detour
            self._blocked_since_s = None
            self._recovery_active = False
            self._record("lateral_detour", {
                "reason": "adaptive_guidance_failed_used_lateral_detour",
                "robot": [robot_x, robot_y, robot_yaw],
                "guidance": list(guidance),
            })

        try:
            candidates = self._runtime.predict_candidates(
                robot_history=list(self._robot_history),
                robot_yaw=robot_yaw,
                human_histories={key: list(value) for key, value in self._human_histories.items()},
                final_goal=goal_xy,
                guidance_point_world=guidance,
            )
        except Exception as exc:  # keep the outdoor control process fail-closed
            self.get_logger().error(f"SPU-BERT inference failed: {exc}")
            self._record("inference_error", {"error": str(exc)})
            self._hold("inference_error")
            return

        attempts = []
        valid_entries = []
        selected: Optional[GuidedInferenceResult] = None
        selected_check = None
        for result in candidates:
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
                    human_histories={key: list(value) for key, value in self._human_histories.items()},
                    human_sample_dt=self.prediction_dt,
                    minimum_human_center_distance=self.minimum_human_center_distance,
                    human_radius=self.human_radius,
                    human_safety_margin=self.human_safety_margin,
                )
                reason = check.reason
            valid = check.valid if check is not None else False
            attempts.append(
                {
                    "rank": int(result.candidate_rank),
                    "candidate_index": int(result.candidate_index),
                    "valid": bool(valid),
                    "reason": reason,
                    "collision": collision_evidence(check),
                    # Keep rejected geometry too; successful logs already contain
                    # the selected path. JSON sanitization is diagnostic-only.
                    "path": [list(point) for point in result.path_world],
                    "goal_progress_m": None if check is None else check.goal_progress_m,
                    "minimum_human_distance_m": (
                        None if check is None or not math.isfinite(check.minimum_human_distance_m)
                        else check.minimum_human_distance_m
                    ),
                }
            )
            if valid:
                valid_entries.append((result, check))

        if valid_entries:
            chosen_index = choose_candidate(
                [list(entry[0].path_world) for entry in valid_entries],
                (robot_x, robot_y), robot_yaw,
                self._last_published_path, now - self._last_published_s,
                self.selection_config,
            )
            selected, selected_check = valid_entries[chosen_index]

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
            self._publish_collision_markers(odom_frame, attempts)
            self._record("prediction", {"valid": False, "reason": reason, "attempts": attempts,
                                        "collision_context": collision_context})
            if self._try_global_path_fallback(
                f"all_candidates_rejected:{reason}", route, goal_xy, robot_x, robot_y,
                robot_yaw, odom_frame, now,
            ):
                return
            self._blocked_hold(
                f"all_candidates_rejected:{reason}", now, robot_x, robot_y, robot_yaw, odom_frame
            )
            return

        self._blocked_since_s = None
        self._recovery_active = False
        self._global_fallback_since_s = None
        path_message = self._path_message(odom_frame, selected.path_world)
        self.path_pub.publish(path_message)
        self._last_published_path = [(robot_x, robot_y), *[tuple(p) for p in selected.path_world]]
        self._last_published_s = now
        self.goal_pub.publish(self._point_pose(odom_frame, goal_xy))
        self._publish_collision_markers(odom_frame, attempts)
        self._status(
            f"path_valid rank={selected.candidate_rank}/{len(candidates)} "
            f"human_min={selected_check.minimum_human_distance_m:.3f}"
        )
        self._record(
            "prediction",
            {
                "valid": True,
                "robot": [robot_x, robot_y],
                "goal": list(goal_xy),
                "guidance": list(guidance),
                "selected_rank": int(selected.candidate_rank),
                "selected_candidate_index": int(selected.candidate_index),
                "path": [list(point) for point in selected.path_world],
                "attempts": attempts,
                "collision_context": collision_context,
            },
        )

    def _request_route(self) -> bool:
        if self._goal is None or self._route_pending:
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
        if generation != self._goal_generation:
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
        if generation != self._goal_generation:
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

    def _publish_collision_markers(self, frame_id: str, attempts: list) -> None:
        """Bounded, expiring markers on a separate topic; never alter control."""
        try:
            array = MarkerArray()
            clear = Marker()
            clear.action = Marker.DELETEALL
            array.markers.append(clear)
            for index, attempt in enumerate(attempts):
                evidence = attempt.get("collision", {})
                point = evidence.get("first_collision_point")
                if point is None:
                    continue
                marker = Marker()
                marker.header.frame_id = frame_id
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "first_collision_robot_center"
                marker.id = index
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.orientation.w = 1.0
                marker.pose.position.x, marker.pose.position.y = float(point[0]), float(point[1])
                marker.pose.position.z = 0.25
                marker.scale.x = marker.scale.y = marker.scale.z = 0.20
                at_start = evidence.get("current_footprint_collision", False)
                marker.color = ColorRGBA(r=1.0, g=0.0, b=1.0 if at_start else 0.0, a=0.95)
                marker.lifetime = Duration(seconds=2.0).to_msg()
                array.markers.append(marker)
                label = Marker()
                label.header = marker.header
                label.ns = "collision_candidate_label"
                label.id = index
                label.type = Marker.TEXT_VIEW_FACING
                label.action = Marker.ADD
                label.pose.position.x, label.pose.position.y = float(point[0]), float(point[1])
                label.pose.position.z = 0.5 + index * 0.15
                label.pose.orientation.w = 1.0
                label.scale.z = 0.14
                label.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
                label.text = f"rank={attempt['rank']} seg={evidence['first_collision_segment_index']}"
                label.lifetime = Duration(seconds=2.0).to_msg()
                array.markers.append(label)
            self.collision_marker_pub.publish(array)
        except Exception as exc:
            self.get_logger().warning(f"Collision marker diagnostics failed: {exc}",
                                      throttle_duration_sec=5.0)

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

    def _record_throttled(self, event: str, payload: dict, now: float, period_s: float = 2.0) -> None:
        if now - self._throttled_record_s.get(event, -math.inf) < period_s:
            return
        self._throttled_record_s[event] = now
        self._record(event, payload)

    def _try_global_path_fallback(
        self,
        reason: str,
        route,
        goal_xy: XY,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        odom_frame: str,
        now: float,
    ) -> bool:
        """Publish a validated fallback path (route prefix, else arc); False = hold.

        Uses validate_candidate_path unchanged, so footprint, pedestrian and
        goal-progress rules are identical to those applied to model output.
        """
        if not self.global_path_fallback or not route:
            return False
        if (
            self._global_fallback_since_s is not None
            and now - self._global_fallback_last_s > 2.0
        ):
            self._global_fallback_since_s = None  # a gap ended the previous episode
        if (
            self._global_fallback_since_s is not None
            and now - self._global_fallback_since_s > self.global_fallback_max_sec
        ):
            self._record_throttled("global_path_fallback_rejected", {
                "trigger": reason, "reason": "episode_time_limit",
                "episode_s": now - self._global_fallback_since_s,
            }, now)
            return False
        count = max(int(self.pred_len), 1)
        spacing = self.global_fallback_length_m / count
        points = route_prefix_points(route, (robot_x, robot_y), count, spacing) or []
        footprint_radius = self.robot_radius + self.static_safety_margin
        minimum_points = max(1, math.ceil(self.global_fallback_min_safe_length_m / spacing))
        human_histories = {key: list(value) for key, value in self._human_histories.items()}

        tier = "normal"
        found, last_reason = self._find_fallback_path(
            points, route, goal_xy, robot_x, robot_y, robot_yaw, footprint_radius,
            human_histories, count, spacing, minimum_points,
        )
        escape_radius = (
            self._adaptive_escape_radius(robot_x, robot_y, footprint_radius)
            if found is None else None
        )
        if found is None and escape_radius is not None:
            # Every path failed at the normal radius, but the robot itself is not
            # touching anything at the reduced one: keep moving (and away) rather
            # than deadlock because the robot already sits inside the 0.44 m margin.
            found, escape_reason = self._find_fallback_path(
                points, route, goal_xy, robot_x, robot_y, robot_yaw, escape_radius,
                human_histories, count, spacing, minimum_points,
            )
            if found is not None:
                tier = "escape"
            else:
                last_reason = f"{last_reason}|escape:{escape_reason}"
        if found is None:
            self._record_throttled("global_path_fallback_rejected", {
                "trigger": reason, "reason": last_reason,
                "route_prefix_points": len(points),
                "minimum_points": minimum_points,
                "clearance_probe": self._clearance_probe(robot_x, robot_y),
            }, now)
            return False
        chosen, chosen_shift, check, source, chosen_curvature, chosen_heading = found
        if self._global_fallback_since_s is None:
            self._global_fallback_since_s = now
        self._global_fallback_last_s = now
        self._blocked_since_s = None
        self._recovery_active = False
        self.path_pub.publish(self._path_message(odom_frame, chosen))
        self._last_published_path = [(robot_x, robot_y), *[tuple(p) for p in chosen]]
        self._last_published_s = now
        self.goal_pub.publish(self._point_pose(odom_frame, goal_xy))
        self._status(
            f"path_valid global_fallback source={source} tier={tier}{'' if tier == 'normal' else f'({escape_radius:.2f}m)'} points={len(chosen)}/{len(points) or len(chosen)} "
            f"shift={chosen_shift:+.2f} curvature={chosen_curvature or 0.0:+.2f} "
            f"heading={chosen_heading or 0.0:+.2f} human_min={check.minimum_human_distance_m:.3f}"
        )
        self._record("global_path_fallback", {
            "trigger": reason,
            "robot": [robot_x, robot_y],
            "goal": list(goal_xy),
            "path": [list(point) for point in chosen],
            "points_kept": len(chosen),
            "points_total": len(points) or len(chosen),
            "lateral_shift_m": chosen_shift,
            "source": source,
            "tier": tier,
            "footprint_radius_m": escape_radius if tier == "escape" else footprint_radius,
            "curvature": chosen_curvature,
            "heading_offset_rad": chosen_heading,
            "episode_s": now - self._global_fallback_since_s,
            "minimum_human_distance_m": (
                None if not math.isfinite(check.minimum_human_distance_m)
                else check.minimum_human_distance_m
            ),
        })
        return True

    def _find_fallback_path(
        self,
        points,
        route,
        goal_xy: XY,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        footprint_radius: float,
        human_histories,
        count: int,
        spacing: float,
        minimum_points: int,
    ):
        """Best validated route-prefix (slid sideways) or arc at ``footprint_radius``.

        Returns ((path, shift, check, source, curvature, heading_offset) | None, last_reason).
        """
        chosen = None
        chosen_shift = 0.0
        check = None
        last_reason = "prefix_too_short"
        keep_options = sorted({len(points), max(minimum_points, (len(points) * 3) // 4), minimum_points},
                              reverse=True)
        for shift in (lateral_offsets(
            self.global_fallback_max_shift_m, self.global_fallback_shift_step_m
        ) if points else []):
            shifted = shift_path_laterally(
                points, (robot_x, robot_y), shift, self.global_fallback_taper_m
            )
            if shifted is None:
                continue
            for keep in keep_options:
                if chosen is not None and keep <= len(chosen):
                    break  # cannot beat the prefix already found
                candidate = shifted[:keep]
                attempt = validate_candidate_path(
                    current=(robot_x, robot_y),
                    path=candidate,
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
                if attempt.valid:
                    chosen, chosen_shift, check = candidate, shift, attempt
                    break
                if shift == 0.0:
                    last_reason = attempt.reason
            if chosen is not None and len(chosen) == len(points):
                break  # full-length prefix at the smallest shift: cannot improve
        if chosen is not None:
            return (chosen, chosen_shift, check, "route_prefix", None, None), last_reason
        if self.arc_planner_fallback:
            arc = self._select_arc_path(
                route, goal_xy, robot_x, robot_y, robot_yaw, footprint_radius,
                human_histories, count,
            )
            if arc is not None:
                pts, curvature, arc_check, heading = arc
                return (pts, 0.0, arc_check, "arc", curvature, heading), last_reason
        return None, last_reason

    def _select_arc_path(
        self,
        route,
        goal_xy: XY,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        footprint_radius: float,
        human_histories,
        count: int,
    ):
        """Best validated constant-curvature arc, or None. (points, curvature, check, heading_offset)."""
        ahead = route_prefix_points(
            route, (robot_x, robot_y), 1, max(self.arc_target_distance_m, 0.1)
        )
        target = ahead[-1] if ahead else (route[-1] if route else goal_xy)
        maximum_length = self.global_fallback_length_m
        best = None
        best_cost = math.inf
        for heading_offset in self.arc_heading_offsets:
            # Off-axis starts use a gentler fan: the pivot already did the turning.
            fan = (
                curvature_fan(self.arc_max_curvature, self.arc_curvature_step)
                if heading_offset == 0.0
                else curvature_fan(min(self.arc_max_curvature, 0.8), 2 * self.arc_curvature_step)
            )
            for length in (maximum_length, min(1.0, maximum_length)):
                for curvature in fan:
                    pts = arc_points(
                        (robot_x, robot_y), robot_yaw + heading_offset, curvature, length, count
                    )
                    if pts is None:
                        continue
                    cost = arc_cost(pts[-1], target, curvature, length, maximum_length,
                                    heading_offset=heading_offset)
                    if cost >= best_cost:
                        continue  # cannot beat the best valid arc: skip the costly check
                    check = validate_candidate_path(
                        current=(robot_x, robot_y),
                        path=pts,
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
                    if check.valid:
                        best, best_cost = (pts, curvature, check, heading_offset), cost
        return best

    def _blocked_hold(
        self,
        reason: str,
        now: float,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        odom_frame: str,
    ) -> None:
        """Hold caused by the guidance point / all candidates being rejected
        as blocked. A brief hold is normal (the block may be a passing
        pedestrian); if it persists, stop just sitting there and sweep-scan
        in place instead, since nothing else in this stack will unstick it.
        """
        if self._blocked_since_s is None:
            self._blocked_since_s = now
        blocked_duration = now - self._blocked_since_s
        if blocked_duration < self.stuck_hold_timeout_sec:
            self._recovery_active = False
            self._hold(reason)
            return
        self._run_recovery_sweep(reason, blocked_duration, now, robot_x, robot_y, robot_yaw, odom_frame)

    def _adaptive_escape_radius(self, robot_x: float, robot_y: float, footprint_radius: float):
        """Largest radius in [floor, min(escape cap, normal)) at which the robot itself is free.

        None when the robot is closer to an obstacle than the floor, or escape is off.
        """
        upper = min(self.escape_footprint_radius_m, footprint_radius - 0.01)
        floor = self.escape_floor_radius_m
        if not (upper > 0.0 and floor > 0.0 and upper >= floor - 1e-9):
            return None
        radius = upper
        while radius >= floor - 1e-9:
            if self.map_provider.path_collision_cost(
                [(robot_x, robot_y)], radius=radius, weight=1.0
            ) <= 0.0:
                return radius
            radius = round(radius - 0.01, 6)
        return None

    def _clearance_probe(self, robot_x: float, robot_y: float) -> dict:
        """Which radii around the robot centre already touch the rolling map.

        Diagnostic only: tells a "no room anywhere" stop apart from a "robot is
        already inside the footprint radius, so every swept path starts in
        collision" deadlock.
        """
        probe = {}
        for radius in (0.30, 0.34, 0.38, 0.44, 0.50):
            try:
                hit = self.map_provider.path_collision_cost(
                    [(robot_x, robot_y)], radius=radius, weight=1.0
                ) > 0.0
            except Exception:  # never let a diagnostic block the stop path
                hit = None
            probe[f"{radius:.2f}"] = hit
        return probe

    def _run_recovery_sweep(
        self,
        reason: str,
        blocked_duration: float,
        now: float,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        odom_frame: str,
    ) -> None:
        if self.recovery_rotation_clearance_m > 0.0 and self.map_provider.path_collision_cost(
            [(robot_x, robot_y)], radius=self.recovery_rotation_clearance_m, weight=1.0
        ) > 0.0:
            self._record_throttled("recovery_sweep_blocked", {
                "reason": reason,
                "blocked_duration_s": blocked_duration,
                "robot": [robot_x, robot_y, robot_yaw],
                "rotation_clearance_m": self.recovery_rotation_clearance_m,
                "clearance_probe": self._clearance_probe(robot_x, robot_y),
            }, now)
            self._hold(f"recovery_blocked_obstacle_in_rotation_circle:{reason}")
            return
        if not self._recovery_active:
            self._recovery_active = True
            self._recovery_started_s = now
            # A fresh global-path request often unsticks a Nav2-side cause
            # once we're clearly stuck, without waiting for the normal
            # route_refresh_period_sec throttle.
            if self.require_global_path and not self._route_pending:
                self._request_route()
        elapsed = now - self._recovery_started_s
        phase = int(elapsed // max(self.recovery_sweep_period_sec, 0.1))
        sign = 1.0 if phase % 2 == 0 else -1.0
        # sweep_yaw is deliberately re-derived from the live robot_yaw every
        # cycle rather than fixed at recovery start: it keeps the heading
        # error pinned above rotate_in_place_angle (0.95 rad, see
        # safe_path_tracker's config) for the whole sweep phase, so the
        # tracker rotates in place instead of creeping forward toward
        # whatever is blocking us. The tracker's own front-sector hard stop
        # (obstacle_stop_distance) still applies underneath this regardless.
        sweep_yaw = robot_yaw + sign * self.recovery_sweep_angle_rad
        target = (
            robot_x + self.recovery_step_m * math.cos(sweep_yaw),
            robot_y + self.recovery_step_m * math.sin(sweep_yaw),
        )
        self.path_pub.publish(self._path_message(odom_frame, [(robot_x, robot_y), target]))
        self._status(
            f"recovery_sweep:{reason}:sign={sign:+.0f}:blocked={blocked_duration:.1f}s"
        )
        self._record(
            "recovery_sweep",
            {
                "reason": reason,
                "blocked_duration_s": blocked_duration,
                "sweep_elapsed_s": elapsed,
                "sign": sign,
                "robot": [robot_x, robot_y, robot_yaw],
                "target": list(target),
            },
        )

    def _hold(self, reason: str, publish_empty: bool = True) -> None:
        self._status(f"hold:{reason}")
        if not publish_empty:
            return
        frame = self._latest_odom.header.frame_id if self._latest_odom is not None else "odom"
        self.path_pub.publish(self._path_message(frame or "odom", []))
        # Retain freshly published rejection evidence briefly; clear stale
        # evidence on sensor/goal/route holds instead of leaving it in RViz.
        if not reason.startswith("all_candidates_rejected:"):
            self._publish_collision_markers(frame or "odom", [])

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
            self._diagnostics_file.write(diagnostic_json(record) + "\n")
        except Exception as exc:
            self.get_logger().warning(f"Bridge diagnostics write failed: {exc}",
                                      throttle_duration_sec=5.0)

    def destroy_node(self):
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
