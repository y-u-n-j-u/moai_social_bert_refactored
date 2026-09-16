#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path as FilePath
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from moai_nav_msgs.msg import Tracks
from nav_msgs.msg import Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool

from .navigation_core import (
    clamp,
    goal_approach_speed_limit,
    lookahead_point,
    nearest_range_in_sector,
    normalize_angle,
    same_frame_id,
    yaw_from_quaternion,
)
from .sensor_freshness import stamp_error


class SafePathTrackerNode(Node):
    """Track short SPU-BERT paths while enforcing real-robot stop conditions."""

    def __init__(self) -> None:
        super().__init__("safe_path_tracker")

        self.odom_topic = self._string_param("odom_topic", "/aft_mapped_to_init")
        self.path_topic = self._string_param("path_topic", "/spu_bert/predicted_path")
        self.goal_topic = self._string_param("goal_topic", "/spu_bert/final_goal_odom")
        self.scan_topic = self._string_param("scan_topic", "/scan")
        self.tracks_topic = self._string_param("tracks_topic", "/ped_tracking")
        self.cmd_vel_topic = self._string_param("cmd_vel_topic", "/spu_bert/cmd_vel_dryrun")
        self.emergency_stop_topic = self._string_param(
            "emergency_stop_topic", "/spu_bert/emergency_stop"
        )
        self.status_topic = self._string_param("status_topic", "/spu_bert/tracker_status")
        self.diagnostics_topic = self._string_param("diagnostics_topic", "/spu_bert/tracker_diagnostics")
        self.diagnostics_path = self._string_param("diagnostics_path", "")
        self.plan_context_topic = self._string_param("plan_context_topic", "/spu_bert/plan_context")
        self.goal_completion_topic = self._string_param("goal_completion_topic", "/spu_bert/goal_completion")
        self.enable_service = self._string_param("enable_service", "/spu_bert/enable_motion")
        self.base_frame = self._string_param("base_frame", "base_link")

        self.motion_enabled = self._bool_param("motion_enabled", False)
        self.require_plan_context = self._bool_param("require_plan_context", True)
        self.check_sensor_header_stamps = self._bool_param("check_sensor_header_stamps", True)
        self.sensor_stamp_future_tolerance = float(
            self.declare_parameter("sensor_stamp_future_tolerance_sec", 0.10).value
        )
        if not math.isfinite(self.sensor_stamp_future_tolerance) or self.sensor_stamp_future_tolerance < 0.0:
            raise ValueError("sensor_stamp_future_tolerance_sec must be nonnegative and finite")
        self.plan_stamp_future_tolerance = float(
            self.declare_parameter("plan_stamp_future_tolerance_sec", 0.10).value
        )
        if not math.isfinite(self.plan_stamp_future_tolerance) or self.plan_stamp_future_tolerance < 0.0:
            raise ValueError("plan_stamp_future_tolerance_sec must be nonnegative and finite")
        self.require_tracks_heartbeat = self._bool_param("require_tracks_heartbeat", True)
        self.control_rate = float(self.declare_parameter("control_rate_hz", 10.0).value)
        self.lookahead_distance = float(self.declare_parameter("lookahead_distance", 0.70).value)
        self.goal_tolerance = float(self.declare_parameter("goal_tolerance", 0.50).value)
        self.goal_slow_distance = float(self.declare_parameter("goal_slow_distance", 1.50).value)
        self.goal_approach_minimum_speed = float(
            self.declare_parameter("goal_approach_minimum_speed", 0.05).value
        )
        if not math.isfinite(self.goal_slow_distance) or self.goal_slow_distance <= self.goal_tolerance:
            raise ValueError("goal_slow_distance must be finite and greater than goal_tolerance")
        if not math.isfinite(self.goal_approach_minimum_speed) or self.goal_approach_minimum_speed <= 0.0:
            raise ValueError("goal_approach_minimum_speed must be positive and finite")
        self.maximum_linear_speed = float(
            self.declare_parameter("maximum_linear_speed", 0.25).value
        )
        self.maximum_angular_speed = float(
            self.declare_parameter("maximum_angular_speed", 0.55).value
        )
        self.maximum_linear_acceleration = float(
            self.declare_parameter("maximum_linear_acceleration", 0.35).value
        )
        self.maximum_angular_acceleration = float(
            self.declare_parameter("maximum_angular_acceleration", 1.20).value
        )
        self.path_timeout = float(self.declare_parameter("path_timeout_sec", 1.20).value)
        self.odom_timeout = float(self.declare_parameter("odom_timeout_sec", 0.50).value)
        self.scan_timeout = float(self.declare_parameter("scan_timeout_sec", 0.50).value)
        self.tracks_timeout = float(self.declare_parameter("tracks_timeout_sec", 1.00).value)
        self.obstacle_stop_distance = float(
            self.declare_parameter("obstacle_stop_distance", 0.75).value
        )
        self.obstacle_slow_distance = float(
            self.declare_parameter("obstacle_slow_distance", 1.40).value
        )
        self.forward_scan_half_angle = float(
            self.declare_parameter("forward_scan_half_angle", 0.70).value
        )
        self.human_stop_distance = float(
            self.declare_parameter("human_stop_distance", 0.90).value
        )
        self.rotate_in_place_angle = float(
            self.declare_parameter("rotate_in_place_angle", 0.95).value
        )
        self.angular_gain = float(self.declare_parameter("angular_gain", 1.6).value)

        self._odom: Optional[Odometry] = None
        self._path: Optional[Path] = None
        self._goal: Optional[PoseStamped] = None
        self._scan: Optional[LaserScan] = None
        self._scan_frame_valid = False
        self._scan_data_valid = False
        self._scan_invalid_reason = "scan_empty"
        self._tracks_frame_valid = False
        self._tracks_data_valid = True
        self._latest_scan_frame = ""
        self._latest_tracks_frame = ""
        self._minimum_human_distance = math.inf
        self._odom_stamp_s = -math.inf
        self._path_stamp_s = -math.inf
        self._scan_stamp_s = -math.inf
        self._tracks_stamp_s = -math.inf
        self._tracks_source_stamp = None
        self._emergency_latched = False
        self._last_command = Twist()
        self._last_status = ""
        self._plan_context = None
        self._plan_context_stamp_s = -math.inf
        self._completed_goal_id = None
        self._diagnostics_file = None
        if self.diagnostics_path:
            try:
                path = FilePath(self.diagnostics_path).expanduser()
                path.parent.mkdir(parents=True, exist_ok=True)
                self._diagnostics_file = path.open("a", encoding="utf-8", buffering=1)
            except OSError as exc:
                self.get_logger().warning(f"Tracker diagnostics file unavailable: {exc}")

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.diagnostics_pub = self.create_publisher(String, self.diagnostics_topic, 10)
        context_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.completion_pub = self.create_publisher(String, self.goal_completion_topic, context_qos)
        self.create_subscription(String, self.plan_context_topic, self._on_plan_context, context_qos)
        self.create_subscription(Odometry, self.odom_topic, self._on_odom, 20)
        self.create_subscription(Path, self.path_topic, self._on_path, 10)
        self.create_subscription(PoseStamped, self.goal_topic, self._on_goal, 10)
        self.create_subscription(LaserScan, self.scan_topic, self._on_scan, qos_profile_sensor_data)
        self.create_subscription(Tracks, self.tracks_topic, self._on_tracks, 10)
        self.create_subscription(Bool, self.emergency_stop_topic, self._on_emergency_stop, 10)
        self.create_service(SetBool, self.enable_service, self._on_enable_motion)
        self.create_timer(1.0 / max(self.control_rate, 1.0), self._control_tick)

        if self.motion_enabled:
            self.get_logger().warning(
                "motion_enabled was true at startup. For outdoor testing, start disarmed and arm "
                "only after the monitor checklist passes."
            )
        self._status("disarmed" if not self.motion_enabled else "armed_waiting_for_inputs")

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
        self._odom = msg
        self._odom_stamp_s = self._now_s()

    def _on_path(self, msg: Path) -> None:
        if self.require_plan_context:
            return  # Visualization topic; execution receives one atomic context.
        if len(msg.poses) < 2:
            self._path = None
            self._path_stamp_s = -math.inf
            self._stop(self._context_stop_reason("planner_hold"))
            return
        self._path = msg
        self._path_stamp_s = self._now_s()

    def _on_goal(self, msg: PoseStamped) -> None:
        self._goal = msg

    @staticmethod
    def _path_stamp_ns(path: Optional[Path]):
        if path is None:
            return None
        return int(path.header.stamp.sec) * 1_000_000_000 + int(path.header.stamp.nanosec)

    def _on_plan_context(self, msg: String) -> None:
        if not self.require_plan_context:
            return
        try:
            context = json.loads(msg.data)
            if not isinstance(context, dict):
                raise ValueError("expected an object")
            goal_id = context.get("goal_id")
            state = context.get("state")
            if not isinstance(goal_id, str) or not goal_id or state not in {"waiting", "hold", "active", "complete"}:
                raise ValueError("invalid goal_id or state")
            frame = context.get("frame_id", "")
            if frame is None and state != "active":
                frame = ""
            if not isinstance(frame, str):
                raise ValueError("invalid frame_id")
            goal = context.get("goal")
            if goal is not None:
                if not isinstance(goal, list) or len(goal) != 2:
                    raise ValueError("invalid goal")
                goal = [float(value) for value in goal]
                if not all(math.isfinite(value) for value in goal):
                    raise ValueError("nonfinite goal")
            stamp = context.get("path_stamp_ns")
            if state == "active" and (
                not frame or goal is None or type(stamp) is not int
                or stamp < 0 or stamp >= (2 ** 31) * 1_000_000_000
            ):
                raise ValueError("active context requires a goal, frame and path stamp")
            points = None
            if state == "active":
                points = context.get("path")
                if not isinstance(points, list) or len(points) < 2:
                    raise ValueError("active context requires at least two path points")
                if any(not isinstance(point, list) or len(point) != 2 for point in points):
                    raise ValueError("invalid path point")
                points = [[float(value) for value in point] for point in points]
                if not all(math.isfinite(value) for point in points for value in point):
                    raise ValueError("nonfinite path")
            context = {
                "goal_id": goal_id, "state": state, "frame_id": frame,
                "goal": goal, "path_stamp_ns": stamp,
                "path": points,
                "reason": str(context.get("reason", "")),
            }
        except (ValueError, TypeError, OverflowError) as exc:
            self._plan_context = None
            self._plan_context_stamp_s = -math.inf
            self._path = None
            self._path_stamp_s = -math.inf
            self._stop("plan_context_invalid")
            self.get_logger().warning(f"Rejected plan context: {exc}")
            return
        previous_id = None if self._plan_context is None else self._plan_context["goal_id"]
        if goal_id != previous_id and goal_id != self._completed_goal_id:
            self._completed_goal_id = None
        self._plan_context = context
        received_s = self._now_s()
        self._plan_context_stamp_s = received_s
        if state == "complete" or goal_id == self._completed_goal_id:
            self._complete_goal(publish=False)
        elif state in {"waiting", "hold"}:
            self._path = None
            self._path_stamp_s = -math.inf
            self._stop(self._context_stop_reason("planner_hold"))
        else:
            # The goal, path and generation become executable in one callback.
            path = Path()
            path.header.frame_id = frame
            path.header.stamp.sec = stamp // 1_000_000_000
            path.header.stamp.nanosec = stamp % 1_000_000_000
            for x, y in points:
                pose = PoseStamped()
                pose.header = path.header
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.orientation.w = 1.0
                path.poses.append(pose)
            self._path = path
            self._path_stamp_s = received_s

    def _context_stop_reason(self, default: str) -> str:
        context = self._plan_context
        if context is not None:
            if context["state"] == "complete" or context["goal_id"] == self._completed_goal_id:
                return "goal_reached_disarmed"
            if context["state"] in {"waiting", "hold"}:
                return f"hold:{context['reason'] or context['state']}"
        return default

    def _complete_goal(self, *, publish: bool, goal_distance=None) -> None:
        context = self._plan_context
        if context is not None:
            self._completed_goal_id = context["goal_id"]
        self.motion_enabled = False
        self._path = None
        self._path_stamp_s = -math.inf
        self._stop("goal_reached_disarmed", goal_distance=goal_distance)
        if publish and context is not None:
            self.completion_pub.publish(String(data=json.dumps({"goal_id": context["goal_id"]})))

    def _on_scan(self, msg: LaserScan) -> None:
        self._scan = msg
        self._scan_stamp_s = self._now_s()
        self._latest_scan_frame = str(msg.header.frame_id)
        self._scan_frame_valid = bool(self._latest_scan_frame) and same_frame_id(
            self._latest_scan_frame, self.base_frame
        )
        self._scan_data_valid = False
        if len(msg.ranges) == 0:
            self._scan_invalid_reason = "scan_empty"
            return
        if not all(math.isfinite(value) for value in (msg.angle_min, msg.angle_increment)):
            self._scan_invalid_reason = "scan_angles_invalid"
            return
        last_angle = msg.angle_min + (len(msg.ranges) - 1) * msg.angle_increment
        if not math.isfinite(last_angle):
            self._scan_invalid_reason = "scan_angles_invalid"
            return
        for index, value in enumerate(msg.ranges):
            angle = msg.angle_min + index * msg.angle_increment
            if abs(normalize_angle(angle)) <= abs(self.forward_scan_half_angle):
                # +Inf is a valid no-return measurement; NaN/-Inf are not.
                if not math.isnan(value) and value >= 0.0:
                    self._scan_data_valid = True
                    break
        self._scan_invalid_reason = "scan_front_sector_invalid"

    def _on_tracks(self, msg: Tracks) -> None:
        self._tracks_stamp_s = self._now_s()
        self._tracks_source_stamp = getattr(msg.header, "stamp", None)
        self._latest_tracks_frame = str(msg.header.frame_id)
        self._tracks_frame_valid = bool(self._latest_tracks_frame) and same_frame_id(
            self._latest_tracks_frame, self.base_frame
        )
        self._tracks_data_valid = True
        if not self._tracks_frame_valid:
            self._minimum_human_distance = math.inf
            return
        if any(
            not math.isfinite(float(value))
            for track in msg.tracks
            for value in (track.pose.position.x, track.pose.position.y)
        ):
            self._tracks_data_valid = False
            self._minimum_human_distance = math.inf
            return
        distances = [
            math.hypot(float(track.pose.position.x), float(track.pose.position.y))
            for track in msg.tracks
        ]
        if not all(math.isfinite(distance) for distance in distances):
            self._tracks_data_valid = False
            self._minimum_human_distance = math.inf
            return
        self._minimum_human_distance = min(distances, default=math.inf)

    def _on_emergency_stop(self, msg: Bool) -> None:
        if bool(msg.data):
            self._emergency_latched = True
            self.motion_enabled = False
            self._stop("emergency_stop_latched")

    def _on_enable_motion(self, request: SetBool.Request, response: SetBool.Response):
        if request.data:
            if self._emergency_latched:
                response.success = False
                response.message = "Emergency stop is latched; call with data=false first to reset."
                return response
            if self._plan_context is not None and self._plan_context["goal_id"] == self._completed_goal_id:
                response.success = False
                response.message = "This goal is complete; provide a new goal before arming."
                return response
            self.motion_enabled = True
            response.success = True
            response.message = "Motion armed. Keep the physical E-stop operator ready."
            self._status("armed_waiting_for_fresh_path")
            return response
        self.motion_enabled = False
        self._emergency_latched = False
        self._stop("disarmed")
        response.success = True
        response.message = "Motion disarmed and emergency latch reset."
        return response

    def _control_tick(self) -> None:
        if not self.motion_enabled:
            self._stop(self._context_stop_reason("disarmed"))
            return
        now = self._now_s()
        if self._odom is None or now - self._odom_stamp_s > self.odom_timeout:
            self._stop("odom_missing_or_stale")
            return
        if not self._sensor_stamp_ready("odom", getattr(self._odom.header, "stamp", None), self.odom_timeout):
            return
        if self._scan is None or now - self._scan_stamp_s > self.scan_timeout:
            self._stop("scan_missing_or_stale")
            return
        if not self._sensor_stamp_ready("scan", getattr(self._scan.header, "stamp", None), self.scan_timeout):
            return
        if not self._scan_data_valid:
            self._stop(self._scan_invalid_reason)
            return
        if not self._scan_frame_valid:
            self._stop(
                f"scan_frame_mismatch:got={self._latest_scan_frame or '<empty>'},"
                f"expected={self.base_frame}"
            )
            return
        if self.require_tracks_heartbeat and now - self._tracks_stamp_s > self.tracks_timeout:
            self._stop("pedestrian_tracker_heartbeat_stale")
            return
        if self.require_tracks_heartbeat and not self._sensor_stamp_ready("tracks", self._tracks_source_stamp, self.tracks_timeout):
            return
        if self.require_tracks_heartbeat and not self._tracks_frame_valid:
            self._stop(
                f"tracks_frame_mismatch:got={self._latest_tracks_frame or '<empty>'},"
                f"expected={self.base_frame}"
            )
            return
        if not self._tracks_data_valid:
            self._stop("tracks_data_invalid")
            return
        if self._minimum_human_distance < self.human_stop_distance:
            self._stop(f"human_too_close:{self._minimum_human_distance:.2f}m")
            return
        context = self._plan_context if self.require_plan_context else None
        if self.require_plan_context:
            if context is None:
                self._stop("plan_context_missing")
                return
            if context["state"] == "complete" or context["goal_id"] == self._completed_goal_id:
                self._complete_goal(publish=False)
                return
            if context["state"] != "active":
                self._stop(self._context_stop_reason("planner_hold"))
                return
            if now - self._plan_context_stamp_s > self.path_timeout:
                self._stop("plan_context_missing_or_stale")
                return
        if self._path is None or now - self._path_stamp_s > self.path_timeout:
            self._stop("predicted_path_missing_or_stale")
            return
        if context is not None and self._path_stamp_ns(self._path) != context["path_stamp_ns"]:
            self._stop("plan_path_context_mismatch")
            return
        if context is not None:
            # A durable message may be received long after it was planned.
            # Receipt freshness alone must never authorize that old plan.
            plan_age = (self.get_clock().now().nanoseconds - context["path_stamp_ns"]) * 1e-9
            if plan_age > self.path_timeout:
                self._stop("plan_stamp_stale")
                return
            if plan_age < -self.plan_stamp_future_tolerance:
                self._stop("plan_stamp_in_future")
                return
        if context is None and self._goal is None:
            self._stop("final_goal_missing")
            return

        odom_frame = self._odom.header.frame_id or "odom"
        path_frame = self._path.header.frame_id or odom_frame
        goal_frame = context["frame_id"] if context is not None else self._goal.header.frame_id or odom_frame
        if path_frame != odom_frame or goal_frame != odom_frame:
            self._stop(f"frame_mismatch:path={path_frame},goal={goal_frame},odom={odom_frame}")
            return

        pose = self._odom.pose.pose
        robot_x = float(pose.position.x)
        robot_y = float(pose.position.y)
        quaternion = tuple(float(value) for value in (
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w,
        ))
        if not all(math.isfinite(value) for value in (robot_x, robot_y, *quaternion)):
            self._stop("nonfinite_robot_pose")
            return
        orientation_norm = math.hypot(*quaternion)
        if not math.isfinite(orientation_norm) or orientation_norm <= 1e-12:
            self._stop("invalid_robot_orientation")
            return
        robot_yaw = yaw_from_quaternion(*quaternion)
        if not math.isfinite(robot_yaw):
            self._stop("nonfinite_robot_pose")
            return
        goal_xy = context["goal"] if context is not None else (
            float(self._goal.pose.position.x), float(self._goal.pose.position.y)
        )
        if not all(math.isfinite(value) for value in goal_xy):
            self._stop("nonfinite_goal")
            return
        goal_distance = math.hypot(goal_xy[0] - robot_x, goal_xy[1] - robot_y)
        if not math.isfinite(goal_distance):
            self._stop("nonfinite_goal_distance")
            return
        if goal_distance <= self.goal_tolerance:
            self._complete_goal(publish=context is not None, goal_distance=goal_distance)
            return

        minimum_obstacle, minimum_bearing = nearest_range_in_sector(
            self._scan.ranges,
            self._scan.angle_min,
            self._scan.angle_increment,
            self.forward_scan_half_angle,
        )
        if minimum_obstacle < self.obstacle_stop_distance:
            self._stop(
                f"obstacle_too_close:{minimum_obstacle:.2f}m",
                minimum_front_distance=minimum_obstacle, minimum_front_bearing=minimum_bearing,
            )
            return

        points = [
            (float(item.pose.position.x), float(item.pose.position.y))
            for item in self._path.poses
        ]
        target = lookahead_point(
            points, (robot_x, robot_y), self.lookahead_distance, require_forward_progress=True,
        )
        if target is None:
            self._stop(
                "path_exhausted", goal_distance=goal_distance,
                minimum_front_distance=minimum_obstacle, minimum_front_bearing=minimum_bearing,
            )
            return
        heading_error = normalize_angle(math.atan2(target[1] - robot_y, target[0] - robot_x) - robot_yaw)
        angular = clamp(
            self.angular_gain * heading_error,
            -self.maximum_angular_speed,
            self.maximum_angular_speed,
        )
        if abs(heading_error) >= self.rotate_in_place_angle:
            linear = 0.0
        else:
            linear = self.maximum_linear_speed * max(math.cos(heading_error), 0.0)
        linear = min(linear, goal_approach_speed_limit(
            goal_distance, self.goal_tolerance, self.goal_slow_distance,
            self.maximum_linear_speed, self.goal_approach_minimum_speed,
        ))
        if minimum_obstacle < self.obstacle_slow_distance:
            denominator = max(self.obstacle_slow_distance - self.obstacle_stop_distance, 1e-3)
            linear *= clamp(
                (minimum_obstacle - self.obstacle_stop_distance) / denominator,
                0.0,
                1.0,
            )
        if not math.isfinite(linear) or not math.isfinite(angular):
            self._stop("nonfinite_requested_command")
            return
        command = self._publish_command(
            linear, angular, lookahead=target, heading_error=heading_error,
            goal_distance=goal_distance,
            minimum_front_distance=minimum_obstacle, minimum_front_bearing=minimum_bearing,
        )
        self._status(
            f"tracking v={linear:.2f} w={angular:.2f} "
            f"applied_v={command.linear.x:.2f} applied_w={command.angular.z:.2f} "
            f"lookahead=({target[0]:.3f},{target[1]:.3f}) heading_error={heading_error:.3f} "
            f"obstacle={minimum_obstacle:.2f} human={self._minimum_human_distance:.2f}"
        )

    def _sensor_stamp_ready(self, name: str, stamp, timeout: float) -> bool:
        if not self.check_sensor_header_stamps:
            return True
        error = stamp_error(
            stamp,
            now_ns=self.get_clock().now().nanoseconds,
            timeout_sec=timeout,
            future_tolerance_sec=self.sensor_stamp_future_tolerance,
        )
        if error:
            self._stop(f"{name}_stamp_{error}")
            return False
        return True

    def _publish_command(self, linear: float, angular: float, **diagnostics) -> Twist:
        if not math.isfinite(linear) or not math.isfinite(angular):
            self._stop("nonfinite_requested_command", **diagnostics)
            return self._last_command
        dt = 1.0 / max(self.control_rate, 1.0)
        maximum_dv = self.maximum_linear_acceleration * dt
        maximum_dw = self.maximum_angular_acceleration * dt
        command = Twist()
        command.linear.x = self._last_command.linear.x + clamp(
            float(linear) - self._last_command.linear.x, -maximum_dv, maximum_dv
        )
        command.angular.z = self._last_command.angular.z + clamp(
            float(angular) - self._last_command.angular.z, -maximum_dw, maximum_dw
        )
        if not math.isfinite(command.linear.x) or not math.isfinite(command.angular.z):
            self._stop("nonfinite_applied_command", **diagnostics)
            return self._last_command
        self.cmd_pub.publish(command)
        self._last_command = command
        self._publish_diagnostics(linear, angular, command, stop_reason=None, **diagnostics)
        return command

    def _stop(self, reason: str, **diagnostics) -> None:
        command = Twist()
        self.cmd_pub.publish(command)
        self._last_command = command
        self._publish_diagnostics(0.0, 0.0, command, stop_reason=reason, **diagnostics)
        self._status(reason)

    def _publish_diagnostics(
        self, requested_linear, requested_angular, command, *, stop_reason,
        lookahead=None, heading_error=None, goal_distance=None,
        minimum_front_distance=None, minimum_front_bearing=None,
    ) -> None:
        def finite(value):
            return float(value) if value is not None and math.isfinite(value) else None

        now = self._now_s()
        context = self._plan_context
        path_stamp_ns = self._path_stamp_ns(self._path)
        record = {
            "time_sec": now,
            "event": "stop" if stop_reason else "tracking",
            "motion_enabled": bool(self.motion_enabled),
            "goal_id": None if context is None else context["goal_id"],
            "plan_state": None if context is None else context["state"],
            "path_stamp_ns": path_stamp_ns,
            "path_header_age_sec": None if path_stamp_ns is None else finite(now - path_stamp_ns * 1e-9),
            "path_age_sec": finite(now - self._path_stamp_s),
            "context_age_sec": finite(now - self._plan_context_stamp_s),
            "requested_linear": finite(requested_linear),
            "requested_angular": finite(requested_angular),
            "applied_linear": finite(command.linear.x),
            "applied_angular": finite(command.angular.z),
            "lookahead": None if lookahead is None else [finite(value) for value in lookahead],
            "heading_error": finite(heading_error),
            "goal_distance": finite(goal_distance),
            "minimum_front_distance_m": finite(minimum_front_distance),
            "minimum_front_bearing_rad": finite(minimum_front_bearing),
            "stop_reason": stop_reason,
        }
        serialized = json.dumps(record, allow_nan=False)
        self.diagnostics_pub.publish(String(data=serialized))
        if self._diagnostics_file is not None:
            try:
                self._diagnostics_file.write(serialized + "\n")
            except OSError as exc:
                self.get_logger().warning(f"Tracker diagnostics write failed: {exc}")
                try:
                    self._diagnostics_file.close()
                except OSError:
                    pass
                self._diagnostics_file = None

    def _status(self, text: str) -> None:
        if text == self._last_status:
            return
        self._last_status = text
        self.status_pub.publish(String(data=text))
        self.get_logger().info(text)

    def destroy_node(self):
        for _ in range(3):
            self.cmd_pub.publish(Twist())
        if self._diagnostics_file is not None:
            self._diagnostics_file.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SafePathTrackerNode()
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
