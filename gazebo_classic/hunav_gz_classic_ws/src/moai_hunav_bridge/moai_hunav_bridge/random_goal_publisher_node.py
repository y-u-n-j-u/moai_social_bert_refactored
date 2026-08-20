#!/usr/bin/env python3
from __future__ import annotations

import math
import random
from typing import Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from hunav_msgs.msg import Agent
from nav2_msgs.action import ComputePathToPose
from nav2_msgs.srv import ManageLifecycleNodes
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_srvs.srv import Trigger


XY = Tuple[float, float]

DEFAULT_WAYPOINT_ROUTES = {
    # Keep robot waiting endpoints away from pedestrian cyclic goals.  The
    # previous centerline endpoints were only 0.55--0.67 m from a human goal,
    # less than the 0.70 m combined radii, so pedestrians correctly stopped
    # instead of entering an unavoidable collision with a stationary PMB2.
    "training_corridor": ((-6.0, 0.0), (6.0, 0.0)),
    "training_doorway": ((-4.0, 0.0), (4.0, 0.0)),
    "training_intersection": ((-6.0, 0.0), (6.0, 0.0)),
    "training_slalom": ((-10.0, 0.0), (10.0, 0.0)),
    "training_open_plaza": (
        (-9.0, 0.0),
        (0.0, 9.0),
        (9.0, 0.0),
        (0.0, -9.0),
    ),
    "training_route_choice": ((-8.2, -1.4), (9.0, -1.4)),
    "training_bottleneck_merge": ((-9.0, 0.0), (9.0, 0.0)),
    "training_outdoor_chicane": ((-11.0, 0.0), (11.0, 0.0)),
    "training_dual_route": ((-9.0, 0.0), (9.0, 0.0)),
}


class RandomGoalPublisherNode(Node):
    """Publish validated waypoint or random /goal_pose collection episodes."""

    def __init__(self) -> None:
        super().__init__("random_goal_publisher")

        self.goal_topic = str(self.declare_parameter("goal_topic", "/goal_pose").value)
        self.map_topic = str(self.declare_parameter("map_topic", "/map").value)
        self.robot_topic = str(self.declare_parameter("robot_topic", "/robot_states").value)
        self.map_frame = str(self.declare_parameter("map_frame", "map").value)
        self.goal_mode = str(self.declare_parameter("goal_mode", "waypoint").value).strip().lower()
        self.environment_name = str(
            self.declare_parameter("environment_name", "training_corridor").value
        ).strip()
        self.waypoint_route_text = str(
            self.declare_parameter("waypoint_route", "").value
        ).strip()
        self.nav_ready_service = str(
            self.declare_parameter(
                "nav_ready_service",
                "/lifecycle_manager_navigation/is_active",
            ).value
        )
        self.nav_manage_service = str(
            self.declare_parameter(
                "nav_manage_service",
                "/lifecycle_manager_navigation/manage_nodes",
            ).value
        )
        self.nav_recovery_delay = float(
            self.declare_parameter("nav_recovery_delay", 3.0).value
        )
        self.nav_recovery_retry_period = float(
            self.declare_parameter("nav_recovery_retry_period", 10.0).value
        )
        self.nav_ready_request_timeout = float(
            self.declare_parameter("nav_ready_request_timeout", 3.0).value
        )
        self.seed = int(self.declare_parameter("seed", -1).value)
        self.min_goal_distance = float(self.declare_parameter("min_goal_distance", 6.0).value)
        self.max_goal_distance = float(self.declare_parameter("max_goal_distance", 20.0).value)
        self.clearance = float(self.declare_parameter("clearance", 0.55).value)
        self.reached_tolerance = float(self.declare_parameter("reached_tolerance", 0.6).value)
        self.startup_delay = float(self.declare_parameter("startup_delay", 5.0).value)
        self.min_episode_duration = float(
            self.declare_parameter("min_episode_duration", 12.0).value
        )
        self.goal_timeout = float(self.declare_parameter("goal_timeout", 60.0).value)
        self.no_progress_timeout = float(
            self.declare_parameter("no_progress_timeout", 15.0).value
        )
        self.progress_radius = float(self.declare_parameter("progress_radius", 0.15).value)
        self.next_goal_delay = float(self.declare_parameter("next_goal_delay", 2.0).value)
        self.max_goals = int(self.declare_parameter("max_goals", 0).value)
        self.max_sampling_attempts = int(
            self.declare_parameter("max_sampling_attempts", 500).value
        )

        self._rng = random.Random(None if self.seed < 0 else self.seed)
        self._waypoints = self._resolve_waypoints()
        self._waypoint_index = 0
        self._path_waypoint_index: Optional[int] = None
        self._map: Optional[np.ndarray] = None
        self._resolution = 0.0
        self._origin_xy: XY = (0.0, 0.0)
        self._origin_yaw = 0.0
        self._robot_xy: Optional[XY] = None
        self._goal_xy: Optional[XY] = None
        self._goal_started_at: Optional[float] = None
        self._progress_anchor_xy: Optional[XY] = None
        self._progress_anchor_at: Optional[float] = None
        self._path_candidate: Optional[XY] = None
        self._path_check_pending = False
        self._next_goal_at = math.inf
        self._published_goals = 0
        self._finished = False
        self._waiting_logged = False
        self._nav_ready = False
        self._nav_ready_request_pending = False
        self._nav_ready_request_started_at: Optional[float] = None
        self._nav_ready_request_future = None
        self._nav_inactive_since: Optional[float] = None
        self._nav_startup_request_pending = False
        self._last_nav_startup_request_at = -math.inf

        map_qos = QoSProfile(depth=1)
        map_qos.reliability = ReliabilityPolicy.RELIABLE
        map_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._map_sub = self.create_subscription(
            OccupancyGrid, self.map_topic, self._on_map, map_qos
        )
        self._robot_sub = self.create_subscription(
            Agent, self.robot_topic, self._on_robot, 10
        )
        self._goal_pub = self.create_publisher(PoseStamped, self.goal_topic, 10)
        self._nav_ready_client = self.create_client(Trigger, self.nav_ready_service)
        self._nav_manage_client = self.create_client(
            ManageLifecycleNodes, self.nav_manage_service
        )
        self._path_client = ActionClient(self, ComputePathToPose, "/compute_path_to_pose")
        self._timer = self.create_timer(0.5, self._on_timer)

        limit = "unlimited" if self.max_goals <= 0 else str(self.max_goals)
        seed_label = "system-random" if self.seed < 0 else str(self.seed)
        if self.goal_mode not in {"waypoint", "waypoints", "fixed", "random"}:
            raise ValueError(
                f"Unsupported goal_mode={self.goal_mode!r}; use waypoint or random"
            )
        if self.goal_mode != "random" and len(self._waypoints) < 2:
            raise ValueError(
                f"Waypoint mode needs at least two points for {self.environment_name!r}"
            )
        route_label = (
            " -> ".join(f"({x:.1f},{y:.1f})" for x, y in self._waypoints)
            if self.goal_mode != "random"
            else f"distance={self.min_goal_distance:.1f}..{self.max_goal_distance:.1f} m"
        )
        self.get_logger().info(
            "Automatic goal publisher enabled: "
            f"mode={self.goal_mode}, environment={self.environment_name}, "
            f"topic={self.goal_topic}, seed={seed_label}, goals={limit}, "
            f"route={route_label}, "
            f"clearance={self.clearance:.2f} m"
        )

    def _resolve_waypoints(self) -> Tuple[XY, ...]:
        if self.waypoint_route_text:
            route = []
            try:
                for item in self.waypoint_route_text.split(";"):
                    x_text, y_text = item.split(",", maxsplit=1)
                    route.append((float(x_text), float(y_text)))
            except ValueError as exc:
                raise ValueError(
                    "waypoint_route must use 'x,y;x,y' format, for example "
                    "'-8.0,0.0;8.0,0.0'"
                ) from exc
            return tuple(route)
        return tuple(DEFAULT_WAYPOINT_ROUTES.get(self.environment_name, ()))

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_map(self, msg: OccupancyGrid) -> None:
        height = int(msg.info.height)
        width = int(msg.info.width)
        if height <= 0 or width <= 0 or len(msg.data) != height * width:
            self.get_logger().error("Received an invalid occupancy grid")
            return
        self._map = np.asarray(msg.data, dtype=np.int16).reshape(height, width)
        self._resolution = float(msg.info.resolution)
        self._origin_xy = (
            float(msg.info.origin.position.x),
            float(msg.info.origin.position.y),
        )
        orientation = msg.info.origin.orientation
        self._origin_yaw = math.atan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
        )
        self._waiting_logged = False

    def _on_robot(self, msg: Agent) -> None:
        self._robot_xy = (
            float(msg.position.position.x),
            float(msg.position.position.y),
        )

    def _on_timer(self) -> None:
        if self._finished:
            return
        if not self._nav_ready:
            self._request_nav_ready()
            return
        if self._map is None or self._robot_xy is None:
            if not self._waiting_logged:
                self.get_logger().info("Waiting for /map and /robot_states...")
                self._waiting_logged = True
            return

        now = self._now()
        if self._goal_xy is not None and self._goal_started_at is not None:
            elapsed = now - self._goal_started_at
            distance = math.dist(self._robot_xy, self._goal_xy)
            reached = distance <= self.reached_tolerance
            timed_out = elapsed >= self.goal_timeout
            stalled = self._goal_has_stalled(now)
            if (reached and elapsed >= self.min_episode_duration) or timed_out or stalled:
                reason = "reached" if reached else ("stalled" if stalled else "timeout")
                self.get_logger().info(
                    f"Goal {self._published_goals} finished: {reason}, "
                    f"elapsed={elapsed:.1f}s, remaining={distance:.2f}m"
                )
                self._goal_xy = None
                self._goal_started_at = None
                self._progress_anchor_xy = None
                self._progress_anchor_at = None
                self._next_goal_at = now + max(self.next_goal_delay, 0.0)

        if (
            self._goal_xy is not None
            or self._path_check_pending
            or now < self._next_goal_at
        ):
            return
        if self.max_goals > 0 and self._published_goals >= self.max_goals:
            self._finished = True
            self.get_logger().info(
                f"Automatic goal publisher completed {self._published_goals} goals"
            )
            return

        candidate = (
            self._sample_goal()
            if self.goal_mode == "random"
            else self._select_waypoint()
        )
        if candidate is None:
            label = "free random goal" if self.goal_mode == "random" else "usable waypoint"
            self.get_logger().warning(f"Could not find a {label}; retrying in 2 seconds")
            self._next_goal_at = now + 2.0
            return
        self._validate_path(candidate)

    def _select_waypoint(self) -> Optional[XY]:
        if not self._waypoints or self._robot_xy is None:
            return None
        for offset in range(len(self._waypoints)):
            index = (self._waypoint_index + offset) % len(self._waypoints)
            candidate = self._waypoints[index]
            if math.dist(self._robot_xy, candidate) > self.reached_tolerance:
                self._path_waypoint_index = index
                return candidate
        return None

    def _request_nav_ready(self) -> None:
        # An action server is created before its lifecycle node is active.
        # Treating server_is_ready() as navigation readiness can leave a path
        # request pending forever when Nav2 bringup stalls during activation.
        now = self._now()
        if self._nav_ready_request_pending:
            started_at = self._nav_ready_request_started_at
            if (
                started_at is not None
                and now - started_at >= max(self.nav_ready_request_timeout, 0.5)
            ):
                future = self._nav_ready_request_future
                if future is not None:
                    future.cancel()
                self._nav_ready_request_pending = False
                self._nav_ready_request_started_at = None
                self._nav_ready_request_future = None
                if self._nav_inactive_since is None:
                    self._nav_inactive_since = now
                self.get_logger().warning(
                    "Nav2 lifecycle readiness request timed out; requesting "
                    "startup recovery instead of sending a path goal early"
                )
                self._request_nav_startup(now)
            return
        if not self._nav_ready_client.service_is_ready():
            if self._nav_inactive_since is None:
                self._nav_inactive_since = now
            if not self._waiting_logged:
                self.get_logger().info(
                    f"Waiting for Nav2 lifecycle service {self.nav_ready_service}..."
                )
                self._waiting_logged = True
            if now - self._nav_inactive_since >= max(self.nav_recovery_delay, 0.0):
                self._request_nav_startup(now)
            return
        self._nav_ready_request_pending = True
        self._nav_ready_request_started_at = now
        future = self._nav_ready_client.call_async(Trigger.Request())
        self._nav_ready_request_future = future
        future.add_done_callback(self._on_nav_ready_response)

    def _on_nav_ready_response(self, future) -> None:
        self._nav_ready_request_pending = False
        self._nav_ready_request_started_at = None
        self._nav_ready_request_future = None
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().warning(f"Nav2 readiness check failed: {exc}")
            return
        if response is None or not bool(response.success):
            now = self._now()
            if self._nav_inactive_since is None:
                self._nav_inactive_since = now
            if now - self._nav_inactive_since >= max(self.nav_recovery_delay, 0.0):
                self._request_nav_startup(now)
            return
        self._set_nav_ready("lifecycle manager is active")

    def _request_nav_startup(self, now: float) -> None:
        """Recover when Nav2's first automatic lifecycle startup races DDS."""
        if self._nav_startup_request_pending:
            return
        if now - self._last_nav_startup_request_at < max(
            self.nav_recovery_retry_period, 0.5
        ):
            return
        if not self._nav_manage_client.service_is_ready():
            return
        self._last_nav_startup_request_at = now
        self._nav_startup_request_pending = True
        request = ManageLifecycleNodes.Request()
        request.command = 0  # STARTUP
        future = self._nav_manage_client.call_async(request)
        future.add_done_callback(self._on_nav_startup_response)

    def _on_nav_startup_response(self, future) -> None:
        self._nav_startup_request_pending = False
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().warning(f"Nav2 lifecycle recovery request failed: {exc}")
            return
        if response is not None and bool(response.success):
            self.get_logger().warning(
                "Nav2 remained inactive after initial bringup; requested lifecycle startup recovery"
            )
        else:
            self.get_logger().warning(
                "Nav2 lifecycle startup recovery was rejected; will retry"
            )

    def _set_nav_ready(self, source: str) -> None:
        if self._nav_ready:
            return
        self._nav_ready = True
        self._nav_inactive_since = None
        self._waiting_logged = False
        self._next_goal_at = self._now() + max(self.startup_delay, 0.0)
        self.get_logger().info(
            f"Nav2 is active ({source}); first automatic goal starts after "
            f"{self.startup_delay:.1f}s"
        )

    def _validate_path(self, candidate: XY) -> None:
        if not self._path_client.server_is_ready():
            self.get_logger().info("Waiting for Nav2 ComputePathToPose server...")
            self._next_goal_at = self._now() + 1.0
            return
        goal = ComputePathToPose.Goal()
        goal.goal = self._make_goal_message(candidate)
        goal.use_start = False
        goal.planner_id = ""
        self._path_candidate = candidate
        self._path_check_pending = True
        future = self._path_client.send_goal_async(goal)
        future.add_done_callback(self._on_path_goal_response)

    def _on_path_goal_response(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().warning(f"Path validation request failed: {exc}")
            self._reject_path_candidate()
            return
        if goal_handle is None or not goal_handle.accepted:
            self._reject_path_candidate()
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_path_result)

    def _on_path_result(self, future) -> None:
        candidate = self._path_candidate
        try:
            wrapped_result = future.result()
            path = wrapped_result.result.path
            has_path = len(path.poses) >= 2
        except Exception as exc:
            self.get_logger().warning(f"Path validation result failed: {exc}")
            has_path = False
        self._path_check_pending = False
        self._path_candidate = None
        if not has_path or candidate is None:
            if candidate is not None:
                self.get_logger().info(
                    f"Rejected unreachable goal candidate "
                    f"({candidate[0]:.2f}, {candidate[1]:.2f})"
                )
            self._path_waypoint_index = None
            self._next_goal_at = self._now() + 0.5
            return
        if self.goal_mode != "random" and self._path_waypoint_index is not None:
            self._waypoint_index = (self._path_waypoint_index + 1) % len(self._waypoints)
        self._path_waypoint_index = None
        self.get_logger().info(
            f"Validated Nav2 path with {len(path.poses)} poses to "
            f"({candidate[0]:.2f}, {candidate[1]:.2f})"
        )
        self._publish_goal(candidate, self._now())

    def _reject_path_candidate(self) -> None:
        self._path_check_pending = False
        self._path_candidate = None
        self._path_waypoint_index = None
        self._next_goal_at = self._now() + 0.5

    def _sample_goal(self) -> Optional[XY]:
        grid = self._map
        robot = self._robot_xy
        if grid is None or robot is None or self._resolution <= 0.0:
            return None

        free_rows, free_cols = np.nonzero(grid == 0)
        if free_rows.size == 0:
            return None
        clearance_cells = max(1, int(math.ceil(self.clearance / self._resolution)))
        height, width = grid.shape

        for _ in range(max(self.max_sampling_attempts, 1)):
            index = self._rng.randrange(free_rows.size)
            row = int(free_rows[index])
            col = int(free_cols[index])
            if (
                row < clearance_cells
                or col < clearance_cells
                or row >= height - clearance_cells
                or col >= width - clearance_cells
            ):
                continue
            neighborhood = grid[
                row - clearance_cells : row + clearance_cells + 1,
                col - clearance_cells : col + clearance_cells + 1,
            ]
            if np.any(neighborhood != 0):
                continue
            candidate = self._grid_to_world(row, col)
            distance = math.dist(robot, candidate)
            if distance < self.min_goal_distance:
                continue
            if self.max_goal_distance > 0.0 and distance > self.max_goal_distance:
                continue
            return candidate
        return None

    def _grid_to_world(self, row: int, col: int) -> XY:
        local_x = (float(col) + 0.5) * self._resolution
        local_y = (float(row) + 0.5) * self._resolution
        cosine = math.cos(self._origin_yaw)
        sine = math.sin(self._origin_yaw)
        return (
            self._origin_xy[0] + cosine * local_x - sine * local_y,
            self._origin_xy[1] + sine * local_x + cosine * local_y,
        )

    def _publish_goal(self, goal_xy: XY, now: float) -> None:
        message = self._make_goal_message(goal_xy)
        self._goal_pub.publish(message)
        self._goal_xy = goal_xy
        self._goal_started_at = now
        self._progress_anchor_xy = self._robot_xy
        self._progress_anchor_at = now
        self._published_goals += 1
        self.get_logger().info(
            f"Published automatic goal {self._published_goals}: "
            f"({goal_xy[0]:.2f}, {goal_xy[1]:.2f})"
        )

    def _goal_has_stalled(self, now: float) -> bool:
        if (
            self.no_progress_timeout <= 0.0
            or self._robot_xy is None
            or self._progress_anchor_xy is None
            or self._progress_anchor_at is None
        ):
            return False
        if math.dist(self._robot_xy, self._progress_anchor_xy) >= max(self.progress_radius, 0.0):
            self._progress_anchor_xy = self._robot_xy
            self._progress_anchor_at = now
            return False
        return now - self._progress_anchor_at >= self.no_progress_timeout

    def _make_goal_message(self, goal_xy: XY) -> PoseStamped:
        robot = self._robot_xy
        yaw = 0.0
        if robot is not None:
            yaw = math.atan2(goal_xy[1] - robot[1], goal_xy[0] - robot[0])
        message = PoseStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.map_frame
        message.pose.position.x = goal_xy[0]
        message.pose.position.y = goal_xy[1]
        message.pose.orientation.z = math.sin(0.5 * yaw)
        message.pose.orientation.w = math.cos(0.5 * yaw)
        return message


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RandomGoalPublisherNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        # ROS 2 Humble can surface SIGTERM as an invalid-context RCLError
        # instead of ExternalShutdownException. Suppress only shutdown errors.
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
