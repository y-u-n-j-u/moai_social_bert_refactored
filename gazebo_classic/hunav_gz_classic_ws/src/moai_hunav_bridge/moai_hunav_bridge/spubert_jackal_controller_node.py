#!/usr/bin/env python3
from __future__ import annotations

from math import atan2, cos, hypot, pi, sin
from typing import Dict, List, Optional, Tuple

import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, PoseStamped, Twist
from hunav_msgs.msg import Agent, Agents
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


class SpubertJackalControllerNode(Node):
    """Use predicted pedestrian futures as a social local planner for Jackal.

    SPU-BERT remains responsible for pedestrian trajectory prediction in the
    HuNav compute node. This node generates lightweight differential-drive
    robot rollouts and scores them against those predicted pedestrian paths.
    """

    def __init__(self) -> None:
        super().__init__("spubert_jackal_controller")

        self.robot_topic = str(self.declare_parameter("robot_topic", "/robot_states").value)
        self.humans_topic = str(self.declare_parameter("humans_topic", "/human_states").value)
        self.goal_topic = str(self.declare_parameter("goal_topic", "/goal_pose").value)
        self.cmd_vel_topic = str(self.declare_parameter("cmd_vel_topic", "/cmd_vel").value)
        self.marker_topic = str(self.declare_parameter("marker_topic", "/moai/spubert_jackal_paths").value)
        self.predicted_humans_topic = str(
            self.declare_parameter("predicted_humans_topic", "/moai/social_bert_predicted_paths").value
        )

        self.pred_len = int(self.declare_parameter("pred_len", 12).value)
        self.prediction_dt = float(self.declare_parameter("prediction_dt", 0.4).value)
        self.control_rate = float(self.declare_parameter("control_rate", 10.0).value)
        self.lookahead_step = int(self.declare_parameter("lookahead_step", 3).value)

        self.max_linear_speed = float(self.declare_parameter("max_linear_speed", 1.05).value)
        self.max_angular_speed = float(self.declare_parameter("max_angular_speed", 1.8).value)
        self.rollout_linear_samples = int(self.declare_parameter("rollout_linear_samples", 2).value)
        self.rollout_angular_samples = int(self.declare_parameter("rollout_angular_samples", 9).value)
        self.rollout_max_angular = float(self.declare_parameter("rollout_max_angular", 1.1).value)
        self.goal_tolerance = float(self.declare_parameter("goal_tolerance", 0.45).value)
        self.robot_radius = float(self.declare_parameter("robot_radius", 0.45).value)
        self.human_clearance = float(self.declare_parameter("human_clearance", 1.0).value)
        self.obstacle_clearance = float(self.declare_parameter("obstacle_clearance", 0.8).value)
        self.goal_progress_weight = float(self.declare_parameter("goal_progress_weight", 2.2).value)
        self.goal_path_weight = float(self.declare_parameter("goal_path_weight", 0.9).value)
        self.turn_cost_weight = float(self.declare_parameter("turn_cost_weight", 0.35).value)

        self._robot: Optional[Agent] = None
        self._humans: Dict[int, Agent] = {}
        self._predicted_human_paths: Dict[int, List[Tuple[float, float]]] = {}
        self._goal: Optional[PoseStamped] = None
        self._candidate_paths: List[List[Tuple[float, float]]] = []
        self._selected_path: List[Tuple[float, float]] = []

        self._robot_sub = self.create_subscription(Agent, self.robot_topic, self._on_robot, 10)
        self._humans_sub = self.create_subscription(Agents, self.humans_topic, self._on_humans, 10)
        self._goal_sub = self.create_subscription(PoseStamped, self.goal_topic, self._on_goal, 10)
        self._predicted_humans_sub = self.create_subscription(
            MarkerArray, self.predicted_humans_topic, self._on_predicted_human_markers, 10
        )
        self._cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self._marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 1)
        self._timer = self.create_timer(1.0 / max(self.control_rate, 0.1), self._on_timer)

        self.get_logger().info(
            f"Prediction-aware Jackal controller publishing {self.cmd_vel_topic}; send goals on {self.goal_topic}"
        )

    def _on_robot(self, msg: Agent) -> None:
        self._robot = msg

    def _on_humans(self, msg: Agents) -> None:
        self._humans = {int(agent.id): agent for agent in msg.agents}

    def _on_goal(self, msg: PoseStamped) -> None:
        self._goal = msg
        self.get_logger().info(f"Received Jackal goal: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")

    def _on_predicted_human_markers(self, msg: MarkerArray) -> None:
        predicted: Dict[int, List[Tuple[float, float]]] = {}
        for marker in msg.markers:
            if marker.action == Marker.DELETEALL or "predicted_path" not in marker.ns:
                continue
            agent_id = self._agent_id_from_ns(marker.ns)
            if agent_id is None or len(marker.points) < 2:
                continue
            predicted[agent_id] = [(float(point.x), float(point.y)) for point in marker.points[1:]]
        if predicted:
            self._predicted_human_paths = predicted

    def _on_timer(self) -> None:
        robot = self._robot
        goal = self._goal

        if robot is None or goal is None:
            self._publish_stop()
            return

        if self._distance_to_goal(robot, goal) <= self.goal_tolerance:
            self._publish_stop()
            return

        candidates = self._generate_robot_rollouts(robot, goal)
        selected = self._select_robot_path(robot, self._humans, self._predicted_human_paths, goal, candidates)
        self._candidate_paths = candidates
        self._selected_path = selected
        self._publish_markers(robot, goal, candidates, selected)
        self._cmd_pub.publish(self._path_to_twist(robot, selected))

    def _select_robot_path(
        self,
        robot: Agent,
        humans: Dict[int, Agent],
        predicted_human_paths: Dict[int, List[Tuple[float, float]]],
        goal: PoseStamped,
        candidates: List[List[Tuple[float, float]]],
    ) -> List[Tuple[float, float]]:
        if not candidates:
            return []

        rx = float(robot.position.position.x)
        ry = float(robot.position.position.y)
        gx = float(goal.pose.position.x)
        gy = float(goal.pose.position.y)
        gdx = gx - rx
        gdy = gy - ry
        gdist = max(hypot(gdx, gdy), 1e-6)
        goal_dir_x = gdx / gdist
        goal_dir_y = gdy / gdist
        obstacles = [(float(obs.x), float(obs.y)) for obs in robot.closest_obs]

        best_score = float("inf")
        best_path = candidates[0]
        for path in candidates:
            if not path:
                continue
            score = 0.0
            final_x, final_y = path[-1]
            score += 0.9 * hypot(final_x - gx, final_y - gy)
            progress = (final_x - rx) * goal_dir_x + (final_y - ry) * goal_dir_y
            score -= self.goal_progress_weight * max(progress, 0.0)
            if progress < 0.0:
                score += 20.0 + 5.0 * abs(progress)

            prev_x = rx
            prev_y = ry
            first_dx = path[0][0] - rx
            first_dy = path[0][1] - ry
            if hypot(first_dx, first_dy) > 1e-4:
                initial_yaw_error = abs(self._angle_diff(atan2(first_dy, first_dx), float(robot.yaw)))
                score += self.turn_cost_weight * initial_yaw_error
            for idx, (px, py) in enumerate(path):
                horizon = self.prediction_dt * (idx + 1)
                weight = 1.0 + 0.08 * idx
                point_progress = (px - rx) * goal_dir_x + (py - ry) * goal_dir_y
                score -= self.goal_path_weight * weight * max(point_progress, 0.0)
                step = hypot(px - prev_x, py - prev_y)
                if step > self.max_linear_speed * self.prediction_dt * 2.2:
                    score += 0.5 * step
                prev_x, prev_y = px, py

                for human in humans.values():
                    predicted = predicted_human_paths.get(int(human.id), [])
                    if predicted:
                        hidx = min(idx, len(predicted) - 1)
                        hx, hy = predicted[hidx]
                    else:
                        hx = float(human.position.position.x) + float(human.velocity.linear.x) * horizon
                        hy = float(human.position.position.y) + float(human.velocity.linear.y) * horizon
                    clearance = hypot(px - hx, py - hy) - self.robot_radius - max(float(human.radius), 0.35)
                    score += 2.8 * weight * self._clearance_penalty(clearance, self.human_clearance, 0.15)

                for ox, oy in obstacles:
                    clearance = hypot(px - ox, py - oy) - self.robot_radius
                    score += 1.8 * weight * self._clearance_penalty(clearance, self.obstacle_clearance, 0.12)

            if score < best_score:
                best_score = score
                best_path = path

        return best_path

    def _generate_robot_rollouts(self, robot: Agent, goal: PoseStamped) -> List[List[Tuple[float, float]]]:
        rx = float(robot.position.position.x)
        ry = float(robot.position.position.y)
        yaw = float(robot.yaw)
        gx = float(goal.pose.position.x)
        gy = float(goal.pose.position.y)
        dist_to_goal = hypot(gx - rx, gy - ry)

        linear_values = [self.max_linear_speed]
        if self.rollout_linear_samples >= 2:
            linear_values.insert(0, 0.65 * self.max_linear_speed)
        if self.rollout_linear_samples >= 3:
            linear_values.insert(0, 0.35 * self.max_linear_speed)

        angular_count = max(3, self.rollout_angular_samples)
        if angular_count % 2 == 0:
            angular_count += 1
        angular_values = []
        for idx in range(angular_count):
            ratio = idx / max(angular_count - 1, 1)
            angular_values.append(-self.rollout_max_angular + 2.0 * self.rollout_max_angular * ratio)

        candidates = [self._direct_goal_path(robot, goal)]
        for linear in linear_values:
            for angular in angular_values:
                x = rx
                y = ry
                theta = yaw
                path: List[Tuple[float, float]] = []
                for step_idx in range(self.pred_len):
                    remaining = hypot(gx - x, gy - y)
                    if remaining <= self.goal_tolerance:
                        path.append((x, y))
                        continue
                    speed = min(linear, remaining / max(self.prediction_dt, 1e-3))
                    theta += angular * self.prediction_dt
                    x += speed * cos(theta) * self.prediction_dt
                    y += speed * sin(theta) * self.prediction_dt
                    path.append((x, y))
                if dist_to_goal > self.goal_tolerance and path:
                    candidates.append(path)
        return candidates

    def _path_to_twist(self, robot: Agent, path: List[Tuple[float, float]]) -> Twist:
        idx = max(0, min(self.lookahead_step - 1, len(path) - 1))
        tx, ty = path[idx]
        rx = float(robot.position.position.x)
        ry = float(robot.position.position.y)
        yaw = float(robot.yaw)
        dx = tx - rx
        dy = ty - ry
        distance = hypot(dx, dy)
        target_yaw = atan2(dy, dx)
        yaw_error = self._angle_diff(target_yaw, yaw)

        cmd = Twist()
        cmd.angular.z = max(-self.max_angular_speed, min(self.max_angular_speed, 1.8 * yaw_error))
        heading_scale = max(0.0, cos(yaw_error))
        cmd.linear.x = min(self.max_linear_speed, 0.7 * distance) * heading_scale
        if abs(yaw_error) > 1.2:
            cmd.linear.x = 0.0
        return cmd

    def _publish_markers(
        self,
        robot: Agent,
        goal: PoseStamped,
        candidates: List[List[Tuple[float, float]]],
        selected: List[Tuple[float, float]],
    ) -> None:
        frame_id = goal.header.frame_id or "map"
        stamp = self.get_clock().now().to_msg()
        z = max(float(robot.position.position.z), 0.05) + 0.35
        markers = MarkerArray()
        delete = Marker()
        delete.header.frame_id = frame_id
        delete.header.stamp = stamp
        delete.action = Marker.DELETEALL
        markers.markers.append(delete)

        safe_candidates_marker = Marker()
        safe_candidates_marker.header.frame_id = frame_id
        safe_candidates_marker.header.stamp = stamp
        safe_candidates_marker.ns = "jackal_rollout_safe_paths"
        safe_candidates_marker.id = 1
        safe_candidates_marker.type = Marker.LINE_LIST
        safe_candidates_marker.action = Marker.ADD
        safe_candidates_marker.pose.orientation.w = 1.0
        safe_candidates_marker.scale.x = 0.035
        safe_candidates_marker.color = ColorRGBA(r=0.0, g=0.25, b=1.0, a=0.48)
        safe_candidates_marker.lifetime = Duration(sec=1)

        unsafe_candidates_marker = Marker()
        unsafe_candidates_marker.header.frame_id = frame_id
        unsafe_candidates_marker.header.stamp = stamp
        unsafe_candidates_marker.ns = "jackal_rollout_unsafe_paths"
        unsafe_candidates_marker.id = 2
        unsafe_candidates_marker.type = Marker.LINE_LIST
        unsafe_candidates_marker.action = Marker.ADD
        unsafe_candidates_marker.pose.orientation.w = 1.0
        unsafe_candidates_marker.scale.x = 0.035
        unsafe_candidates_marker.color = ColorRGBA(r=1.0, g=0.05, b=0.0, a=0.55)
        unsafe_candidates_marker.lifetime = Duration(sec=1)

        goals_marker = Marker()
        goals_marker.header.frame_id = frame_id
        goals_marker.header.stamp = stamp
        goals_marker.ns = "jackal_spubert_candidate_goals"
        goals_marker.id = 3
        goals_marker.type = Marker.SPHERE_LIST
        goals_marker.action = Marker.ADD
        goals_marker.pose.orientation.w = 1.0
        goals_marker.scale.x = 0.18
        goals_marker.scale.y = 0.18
        goals_marker.scale.z = 0.18
        goals_marker.color = ColorRGBA(r=0.0, g=0.25, b=1.0, a=0.85)
        goals_marker.lifetime = Duration(sec=1)

        start = Point(x=float(robot.position.position.x), y=float(robot.position.position.y), z=z)
        for path in candidates:
            if not path:
                continue
            target_marker = safe_candidates_marker
            if self._path_min_human_clearance(path, self._humans, self._predicted_human_paths) < 0.15:
                target_marker = unsafe_candidates_marker
            prev = start
            for px, py in path:
                cur = Point(x=float(px), y=float(py), z=z)
                target_marker.points.extend([prev, cur])
                prev = cur
            goals_marker.points.append(Point(x=float(path[-1][0]), y=float(path[-1][1]), z=z + 0.08))

        selected_marker = Marker()
        selected_marker.header.frame_id = frame_id
        selected_marker.header.stamp = stamp
        selected_marker.ns = "jackal_spubert_selected_path"
        selected_marker.id = 4
        selected_marker.type = Marker.LINE_STRIP
        selected_marker.action = Marker.ADD
        selected_marker.pose.orientation.w = 1.0
        selected_marker.scale.x = 0.08
        selected_marker.color = ColorRGBA(r=1.0, g=0.65, b=0.0, a=0.95)
        selected_marker.lifetime = Duration(sec=1)
        selected_marker.points = [start, *[Point(x=x, y=y, z=z + 0.08) for x, y in selected]]

        user_goal_marker = Marker()
        user_goal_marker.header.frame_id = frame_id
        user_goal_marker.header.stamp = stamp
        user_goal_marker.ns = "jackal_user_goal"
        user_goal_marker.id = 5
        user_goal_marker.type = Marker.SPHERE
        user_goal_marker.action = Marker.ADD
        user_goal_marker.pose.position.x = float(goal.pose.position.x)
        user_goal_marker.pose.position.y = float(goal.pose.position.y)
        user_goal_marker.pose.position.z = z
        user_goal_marker.pose.orientation.w = 1.0
        user_goal_marker.scale.x = 0.35
        user_goal_marker.scale.y = 0.35
        user_goal_marker.scale.z = 0.35
        user_goal_marker.color = ColorRGBA(r=0.15, g=1.0, b=0.25, a=0.95)
        user_goal_marker.lifetime = Duration(sec=1)

        markers.markers.extend([safe_candidates_marker, unsafe_candidates_marker, goals_marker, selected_marker, user_goal_marker])
        self._marker_pub.publish(markers)

    def _direct_goal_path(self, robot: Agent, goal: PoseStamped) -> List[Tuple[float, float]]:
        rx = float(robot.position.position.x)
        ry = float(robot.position.position.y)
        gx = float(goal.pose.position.x)
        gy = float(goal.pose.position.y)
        dx = gx - rx
        dy = gy - ry
        dist = max(hypot(dx, dy), 1e-6)
        step = self.max_linear_speed * self.prediction_dt
        path = []
        for idx in range(1, self.pred_len + 1):
            travel = min(dist, step * idx)
            path.append((rx + dx * travel / dist, ry + dy * travel / dist))
        return path

    def _distance_to_goal(self, robot: Agent, goal: PoseStamped) -> float:
        return hypot(
            float(goal.pose.position.x) - float(robot.position.position.x),
            float(goal.pose.position.y) - float(robot.position.position.y),
        )

    def _publish_stop(self) -> None:
        self._cmd_pub.publish(Twist())

    @staticmethod
    def _clearance_penalty(clearance: float, soft: float, hard: float) -> float:
        if clearance >= soft:
            return 0.0
        if clearance <= hard:
            return 100.0 + 80.0 * (hard - clearance)
        return ((soft - clearance) / max(soft - hard, 1e-6)) ** 2

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        diff = a - b
        while diff > pi:
            diff -= 2.0 * pi
        while diff < -pi:
            diff += 2.0 * pi
        return diff

    def _path_min_human_clearance(
        self,
        path: List[Tuple[float, float]],
        humans: Dict[int, Agent],
        predicted_human_paths: Dict[int, List[Tuple[float, float]]],
    ) -> float:
        min_clearance = float("inf")
        for idx, (px, py) in enumerate(path):
            horizon = self.prediction_dt * (idx + 1)
            for human in humans.values():
                predicted = predicted_human_paths.get(int(human.id), [])
                if predicted:
                    hidx = min(idx, len(predicted) - 1)
                    hx, hy = predicted[hidx]
                else:
                    hx = float(human.position.position.x) + float(human.velocity.linear.x) * horizon
                    hy = float(human.position.position.y) + float(human.velocity.linear.y) * horizon
                clearance = hypot(px - hx, py - hy) - self.robot_radius - max(float(human.radius), 0.35)
                min_clearance = min(min_clearance, clearance)
        return min_clearance

    @staticmethod
    def _agent_id_from_ns(ns: str) -> Optional[int]:
        try:
            # Expected form: agent_3_predicted_path
            return int(ns.split("_")[1])
        except (IndexError, ValueError):
            return None


def main(args: List[str] | None = None) -> None:
    rclpy.init(args=args)
    node = SpubertJackalControllerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
