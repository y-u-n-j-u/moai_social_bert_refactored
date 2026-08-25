#!/usr/bin/env python3
from __future__ import annotations

import math
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

from .navigation_core import (
    clamp,
    finite_ranges_in_sector,
    lookahead_point,
    normalize_angle,
    same_frame_id,
    yaw_from_quaternion,
)


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

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)
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
        if len(msg.poses) < 2:
            self._path = None
            self._path_stamp_s = -math.inf
            self._stop("planner_hold")
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
        if minimum_obstacle < self.obstacle_stop_distance:
            self._stop(f"obstacle_too_close:{minimum_obstacle:.2f}m")
            return

        points = [
            (float(item.pose.position.x), float(item.pose.position.y))
            for item in self._path.poses
        ]
        target = lookahead_point(points, (robot_x, robot_y), self.lookahead_distance)
        if target is None:
            self._stop("path_has_no_lookahead")
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
        if minimum_obstacle < self.obstacle_slow_distance:
            denominator = max(self.obstacle_slow_distance - self.obstacle_stop_distance, 1e-3)
            linear *= clamp(
                (minimum_obstacle - self.obstacle_stop_distance) / denominator,
                0.0,
                1.0,
            )
        self._publish_command(linear, angular)
        self._status(
            f"tracking v={linear:.2f} w={angular:.2f} "
            f"obstacle={minimum_obstacle:.2f} human={self._minimum_human_distance:.2f}"
        )

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
