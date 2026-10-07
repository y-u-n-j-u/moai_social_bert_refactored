#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import asdict
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from moai_nav_msgs.msg import Tracks
from nav_msgs.msg import Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool
from visualization_msgs.msg import Marker

from .navigation_core import (
    clamp,
    corridor_obstacle_distance,
    finite_ranges_in_sector,
    PathProgressTracker,
    normalize_angle,
    same_frame_id,
    yaw_from_quaternion,
)
from .obstacle_steering import (
    GapConfig, SteeringConfig, avoidance_steering, combine_with_path_turn, gap_heading,
)
from .tracker_diagnostics import StopEpisode, diagnostic_json


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
        self.diagnostics_topic = self._string_param(
            "diagnostics_topic", "/spu_bert/tracker_diagnostics"
        )
        self.lookahead_marker_topic = self._string_param(
            "lookahead_marker_topic", "/spu_bert/tracker_lookahead"
        )
        self.enable_service = self._string_param("enable_service", "/spu_bert/enable_motion")
        self.base_frame = self._string_param("base_frame", "base_link")

        self.motion_enabled = self._bool_param("motion_enabled", False)
        self.require_tracks_heartbeat = self._bool_param("require_tracks_heartbeat", True)
        self.control_rate = float(self.declare_parameter("control_rate_hz", 10.0).value)
        self.lookahead_distance = float(self.declare_parameter("lookahead_distance", 0.70).value)
        self.goal_tolerance = float(self.declare_parameter("goal_tolerance", 0.50).value)
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
        # Corridor mode (off when half width is 0): stop/slow only for obstacles
        # in the strip the body sweeps when driving straight, so a box edge
        # beside the robot in a narrow passage no longer freezes it. Anything
        # closer than the hard-stop distance anywhere in the forward sector
        # still stops it.
        self.obstacle_corridor_half_width = float(
            self.declare_parameter("obstacle_corridor_half_width", 0.0).value
        )
        self.obstacle_hard_stop_distance = float(
            self.declare_parameter("obstacle_hard_stop_distance", 0.0).value
        )
        # Continuous sideways steering away from obstacles ahead (0 gain = off). It is added to the
        # path-following turn so the robot curves around an obstacle from a distance.
        self.avoidance = SteeringConfig(
            lookahead_m=float(self.declare_parameter("avoidance_lookahead_m", 3.0).value),
            half_width_m=float(self.declare_parameter("avoidance_half_width_m", 0.75).value),
            gain=float(self.declare_parameter("avoidance_gain", 0.0).value),
            max_rate=float(self.declare_parameter("avoidance_max_rate", 0.40).value),
        )
        # Follow-the-gap (0 min width = off): when something blocks the straight line, head for the
        # side of it that has room to pass instead of following a path that may lead into a dead end.
        self.gap = GapConfig(
            horizon_m=float(self.declare_parameter("gap_horizon_m", 3.5).value),
            block_horizon_m=float(self.declare_parameter("gap_block_horizon_m", 3.0).value),
            min_gap_m=float(self.declare_parameter("gap_min_width_m", 0.0).value),
            edge_margin_m=float(self.declare_parameter("gap_edge_margin_m", 0.65).value),
            straight_half_m=float(self.declare_parameter("gap_straight_half_width_m", 0.60).value),
        )
        self._avoid_side = 0
        self._avoid_side_s = -1e9
        self.slow_along_travel_direction = bool(
            self.declare_parameter("slow_along_travel_direction", False).value
        )
        self.human_stop_distance = float(
            self.declare_parameter("human_stop_distance", 0.90).value
        )
        self.rotate_in_place_angle = float(
            self.declare_parameter("rotate_in_place_angle", 0.95).value
        )
        # Forward speed kept while turning toward a target that is far off the heading
        # (below the rotate-in-place angle): drive a curve instead of slowing to a crawl.
        # 0 = off (speed is vmax*cos(error) only). Obstacle slow-down still applies after.
        self.minimum_turn_speed = float(
            self.declare_parameter("minimum_turn_speed", 0.0).value
        )
        self.angular_gain = float(self.declare_parameter("angular_gain", 1.6).value)

        self._odom: Optional[Odometry] = None
        self._path: Optional[Path] = None
        self._goal: Optional[PoseStamped] = None
        self._scan: Optional[LaserScan] = None
        self._scan_frame_valid = False
        self._tracks_frame_valid = False
        self._latest_scan_frame = ""
        self._latest_tracks_frame = ""
        self._minimum_human_distance = math.inf
        self._odom_stamp_s = -math.inf
        self._path_stamp_s = -math.inf
        self._scan_stamp_s = -math.inf
        self._tracks_stamp_s = -math.inf
        self._emergency_latched = False
        self._last_command = Twist()
        self._last_status = ""
        self._progress_tracker = PathProgressTracker()
        self._path_receipt_id = 0
        self._path_received_s = -math.inf
        self._path_source_stamp_ns = 0
        self._tracking_debug = None
        self._stop_episode = StopEpisode()

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.diagnostics_pub = self.create_publisher(String, self.diagnostics_topic, 10)
        self.lookahead_pub = self.create_publisher(Marker, self.lookahead_marker_topic, 10)
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
        self._path_receipt_id += 1
        self._path_received_s = self._now_s()
        self._path_source_stamp_ns = self._header_stamp_ns(msg)
        points = [(float(item.pose.position.x), float(item.pose.position.y)) for item in msg.poses]
        self._progress_tracker.set_path(points, msg.header.frame_id)
        self._tracking_debug = None
        if len(msg.poses) < 2:
            self._path = None
            self._path_stamp_s = -math.inf
            self._stop("planner_hold")
            return
        if not all(math.isfinite(value) for point in points for value in point):
            self._path = None
            self._path_stamp_s = -math.inf
            self._stop("invalid_path_coordinates")
            return
        self._path = msg
        self._path_stamp_s = self._now_s()

    def _on_goal(self, msg: PoseStamped) -> None:
        self._goal = msg

    def _on_scan(self, msg: LaserScan) -> None:
        self._scan = msg
        self._scan_stamp_s = self._now_s()
        self._latest_scan_frame = str(msg.header.frame_id)
        self._scan_frame_valid = bool(self._latest_scan_frame) and same_frame_id(
            self._latest_scan_frame, self.base_frame
        )

    def _on_tracks(self, msg: Tracks) -> None:
        self._tracks_stamp_s = self._now_s()
        self._latest_tracks_frame = str(msg.header.frame_id)
        self._tracks_frame_valid = bool(self._latest_tracks_frame) and same_frame_id(
            self._latest_tracks_frame, self.base_frame
        )
        if not self._tracks_frame_valid:
            self._minimum_human_distance = math.inf
            return
        distances = [
            math.hypot(float(track.pose.position.x), float(track.pose.position.y))
            for track in msg.tracks
        ]
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
            self.motion_enabled = True
            self._stop_episode.finish(self._now_s())
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
            self._stop("disarmed")
            return
        now = self._now_s()
        if self._odom is None or now - self._odom_stamp_s > self.odom_timeout:
            self._stop("odom_missing_or_stale")
            return
        if self._scan is None or now - self._scan_stamp_s > self.scan_timeout:
            self._stop("scan_missing_or_stale")
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
        if self.require_tracks_heartbeat and not self._tracks_frame_valid:
            self._stop(
                f"tracks_frame_mismatch:got={self._latest_tracks_frame or '<empty>'},"
                f"expected={self.base_frame}"
            )
            return
        if self._minimum_human_distance < self.human_stop_distance:
            self._stop(f"human_too_close:{self._minimum_human_distance:.2f}m")
            return
        if self._path is None or now - self._path_stamp_s > self.path_timeout:
            self._stop("predicted_path_missing_or_stale")
            return
        if self._goal is None:
            self._stop("final_goal_missing")
            return

        odom_frame = self._odom.header.frame_id or "odom"
        path_frame = self._path.header.frame_id or odom_frame
        goal_frame = self._goal.header.frame_id or odom_frame
        if path_frame != odom_frame or goal_frame != odom_frame:
            self._stop(f"frame_mismatch:path={path_frame},goal={goal_frame},odom={odom_frame}")
            return

        pose = self._odom.pose.pose
        robot_x = float(pose.position.x)
        robot_y = float(pose.position.y)
        robot_yaw = yaw_from_quaternion(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        if not all(math.isfinite(value) for value in (robot_x, robot_y, robot_yaw)):
            self._stop("invalid_robot_pose")
            return
        goal_distance = math.hypot(
            float(self._goal.pose.position.x) - robot_x,
            float(self._goal.pose.position.y) - robot_y,
        )
        if goal_distance <= self.goal_tolerance:
            self.motion_enabled = False
            self._stop("goal_reached_disarmed")
            return

        front_ranges = finite_ranges_in_sector(
            self._scan.ranges,
            self._scan.angle_min,
            self._scan.angle_increment,
            self.forward_scan_half_angle,
        )
        minimum_obstacle = min(front_ranges, default=math.inf)
        if self.obstacle_corridor_half_width > 0.0:
            if minimum_obstacle < self.obstacle_hard_stop_distance:
                self._stop(f"obstacle_too_close:{minimum_obstacle:.2f}m")
                return
            minimum_obstacle = corridor_obstacle_distance(
                self._scan.ranges,
                self._scan.angle_min,
                self._scan.angle_increment,
                self.obstacle_corridor_half_width,
            )
        if minimum_obstacle < self.obstacle_stop_distance:
            self._stop(f"obstacle_too_close:{minimum_obstacle:.2f}m")
            return

        result = self._progress_tracker.update((robot_x, robot_y), self.lookahead_distance)
        if result is None:
            self._stop("path_has_no_lookahead")
            return
        target = result.target
        if (result.progress_m >= result.path_length_m - 1e-6
                or math.hypot(target[0] - robot_x, target[1] - robot_y) <= 1e-6):
            # An exhausted short path must not command forward via atan2(0, 0),
            # or steer backwards toward an endpoint already passed.
            self._stop("path_endpoint_reached")
            return
        heading_error = normalize_angle(math.atan2(target[1] - robot_y, target[0] - robot_x) - robot_yaw)
        gap_heading_used = None
        if self.gap.active:
            gap = gap_heading(
                self._scan.ranges, self._scan.angle_min, self._scan.angle_increment, self.gap,
                prefer_side=self._avoid_side if now - self._avoid_side_s < 2.0 else 0,
            )
            if gap.blocked and gap.heading is not None:
                heading_error = gap.heading
                gap_heading_used = gap.heading
                self._avoid_side, self._avoid_side_s = gap.side, now
        angular = clamp(
            self.angular_gain * heading_error,
            -self.maximum_angular_speed,
            self.maximum_angular_speed,
        )
        avoid_rate = 0.0
        if (self.avoidance.active and gap_heading_used is None
                and abs(heading_error) < self.rotate_in_place_angle):
            # an obstacle dead ahead: keep last side for a while, else go to the side the path leans
            if now - self._avoid_side_s < 1.5 and self._avoid_side != 0:
                prefer = self._avoid_side
            else:
                prefer = 1 if heading_error >= 0.0 else -1
            steering = avoidance_steering(
                self._scan.ranges, self._scan.angle_min, self._scan.angle_increment,
                self.avoidance, prefer_side=prefer,
            )
            if steering.side != 0:
                self._avoid_side, self._avoid_side_s = steering.side, now
                combined = clamp(
                    combine_with_path_turn(angular, steering),
                    -self.maximum_angular_speed, self.maximum_angular_speed,
                )
                avoid_rate = combined - angular
                angular = combined
        if abs(heading_error) >= self.rotate_in_place_angle:
            linear = 0.0
        else:
            linear = self.maximum_linear_speed * max(math.cos(heading_error), 0.0)
            if self.minimum_turn_speed > 0.0:
                linear = max(linear, min(self.minimum_turn_speed, self.maximum_linear_speed))
        slow_obstacle = minimum_obstacle
        if self.obstacle_corridor_half_width > 0.0 and self.slow_along_travel_direction:
            # Slow for what lies in the direction the robot is going to move, not for an
            # obstacle straight ahead that the path is already turning away from (the stop
            # checks above still use the straight-ahead corridor and the sector hard stop).
            slow_obstacle = corridor_obstacle_distance(
                self._scan.ranges, self._scan.angle_min, self._scan.angle_increment,
                self.obstacle_corridor_half_width,
                axis_angle=clamp(heading_error, -0.9, 0.9),
            )
        if slow_obstacle < self.obstacle_slow_distance:
            denominator = max(self.obstacle_slow_distance - self.obstacle_stop_distance, 1e-3)
            linear *= clamp(
                (slow_obstacle - self.obstacle_stop_distance) / denominator,
                0.0,
                1.0,
            )
        self._tracking_debug = {
            **asdict(result),
            "heading_error_rad": heading_error,
            "goal_distance_m": goal_distance,
            "front_obstacle_distance_m": minimum_obstacle,
        }
        self._publish_command(linear, angular)
        status = (
            f"tracking v={linear:.2f} w={angular:.2f} "
            f"obstacle={minimum_obstacle:.2f} human={self._minimum_human_distance:.2f}"
            f"{f' avoid={avoid_rate:+.2f}' if avoid_rate else ''}"
            f"{f' gap={gap_heading_used:+.2f}' if gap_heading_used is not None else ''}"
        )
        self._status(status)
        self._emit_diagnostics(status, linear, angular, stopped=False)

    def _publish_command(self, linear: float, angular: float) -> None:
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
        self.cmd_pub.publish(command)
        self._last_command = command

    def _stop(self, reason: str) -> None:
        command = Twist()
        self.cmd_pub.publish(command)
        self._last_command = command
        self._status(reason)
        self._tracking_debug = None
        self._emit_diagnostics(reason, 0.0, 0.0, stopped=True)

    @staticmethod
    def _header_stamp_ns(message) -> int:
        if message is None:
            return 0
        return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)

    def _emit_diagnostics(self, status: str, linear: float, angular: float, *, stopped: bool) -> None:
        # Motion has already been published. Diagnostic failures must not skip
        # the safety decision or turn this optional observer into a control path.
        try:
            now = self._now_s()

            def input_age(message, received_s):
                stamp = self._header_stamp_ns(message)
                return {
                    "stamp_ns": stamp or None,
                    "source_age_s": now - stamp * 1e-9 if stamp else None,
                    "received_age_s": now - received_s,
                }

            odom = self._odom
            robot = None
            twist = None
            if odom is not None:
                pose = odom.pose.pose
                q = pose.orientation
                robot = {
                    "frame_id": odom.header.frame_id,
                    "x": pose.position.x, "y": pose.position.y,
                    "yaw_rad": yaw_from_quaternion(q.x, q.y, q.z, q.w),
                }
                twist = {"v": odom.twist.twist.linear.x, "w": odom.twist.twist.angular.z}
            episode = self._stop_episode.observe(status, now, stopped)
            record = {
                "schema_version": 1,
                "event": "stop" if stopped else "tracking",
                "stamp_ns": int(self.get_clock().now().nanoseconds),
                "status": status,
                "motion_enabled": self.motion_enabled,
                "emergency_latched": self._emergency_latched,
                "cmd_vel_topic": self.cmd_vel_topic,
                "path": {
                    "geometry_id": self._progress_tracker.version,
                    "receipt_id": self._path_receipt_id,
                    "frame_id": self._progress_tracker.frame_id,
                    "point_count": len(self._progress_tracker.points),
                    "present": self._path is not None,
                    "fresh": self._path is not None and now - self._path_stamp_s <= self.path_timeout,
                    "stamp_ns": self._path_source_stamp_ns or None,
                    "source_age_s": (now - self._path_source_stamp_ns * 1e-9
                                     if self._path_source_stamp_ns else None),
                    "received_age_s": now - self._path_received_s,
                },
                "robot": robot,
                "lookahead": self._tracking_debug,
                "target_command": {"v": linear, "w": angular},
                "published_command": {"v": self._last_command.linear.x,
                                      "w": self._last_command.angular.z},
                "odom_reported_twist": twist,
                "inputs": {
                    "odom": input_age(self._odom, self._odom_stamp_s),
                    "scan": input_age(self._scan, self._scan_stamp_s),
                    "tracks_received_age_s": now - self._tracks_stamp_s,
                },
                "minimum_human_distance_m": self._minimum_human_distance,
                "stop_episode": episode,
            }
            self.diagnostics_pub.publish(String(data=diagnostic_json(record)))
            marker = Marker()
            marker.header.frame_id = odom.header.frame_id if odom is not None else self.base_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "tracker_lookahead"
            marker.id = 0
            if stopped or self._tracking_debug is None:
                marker.action = Marker.DELETE
            else:
                marker.action = Marker.ADD
                marker.type = Marker.SPHERE
                marker.pose.position.x, marker.pose.position.y = self._tracking_debug["target"]
                marker.pose.position.z = 0.25
                marker.pose.orientation.w = 1.0
                marker.scale.x = marker.scale.y = marker.scale.z = 0.15
                marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1.0, 0.5, 0.0, 1.0
                marker.lifetime.nanosec = 500_000_000
            self.lookahead_pub.publish(marker)
        except Exception as exc:
            self.get_logger().warning(f"Tracker diagnostics unavailable: {exc}", throttle_duration_sec=5.0)

    def _status(self, text: str) -> None:
        if text == self._last_status:
            return
        self._last_status = text
        self.status_pub.publish(String(data=text))
        self.get_logger().info(text)

    def destroy_node(self):
        for _ in range(3):
            self.cmd_pub.publish(Twist())
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
