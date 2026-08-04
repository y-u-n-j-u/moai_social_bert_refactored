#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from hunav_msgs.msg import Agent, Agents
from nav2_msgs.action import FollowPath, NavigateToPose
from nav_msgs.msg import Path
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import ColorRGBA, String
from visualization_msgs.msg import Marker, MarkerArray

from .guided_spubert_runtime import (
    GuidedInferenceResult,
    GuidedSpubertRuntime,
    guidance_point,
)
from .social_bert_compute_agents_node import OccupancyMapProvider


XY = Tuple[float, float]


class SpubertNav2BridgeNode(Node):
    """Connect an RViz goal and guided SPU-BERT path to the Nav2 controller."""

    VALID_MODES = {"nav2", "monitor", "spubert"}

    def __init__(self) -> None:
        super().__init__("spubert_nav2_bridge")

        self.execution_mode = str(
            self.declare_parameter("execution_mode", "nav2").value
        ).strip().lower()
        if self.execution_mode not in self.VALID_MODES:
            raise ValueError(
                f"execution_mode must be one of {sorted(self.VALID_MODES)}, "
                f"got {self.execution_mode!r}"
            )

        self.robot_topic = str(self.declare_parameter("robot_topic", "/robot_states").value)
        self.humans_topic = str(self.declare_parameter("humans_topic", "/human_states").value)
        self.goal_topic = str(self.declare_parameter("goal_topic", "/goal_pose").value)
        self.predicted_humans_topic = str(
            self.declare_parameter(
                "predicted_humans_topic",
                "/moai/social_bert_predicted_paths",
            ).value
        )
        self.path_topic = str(
            self.declare_parameter("path_topic", "/moai/spubert_robot_path").value
        )
        self.marker_topic = str(
            self.declare_parameter(
                "marker_topic",
                "/moai/spubert_robot_path_markers",
            ).value
        )
        self.status_topic = str(
            self.declare_parameter(
                "status_topic",
                "/moai/spubert_robot_planner_status",
            ).value
        )
        diagnostics_path = str(
            self.declare_parameter("diagnostics_path", "").value
        ).strip()
        self.diagnostics_path = (
            os.path.abspath(os.path.expanduser(diagnostics_path))
            if diagnostics_path else ""
        )
        self.follow_path_action = str(
            self.declare_parameter("follow_path_action", "/follow_path").value
        )
        self.navigate_to_pose_action = str(
            self.declare_parameter(
                "navigate_to_pose_action",
                "/navigate_to_pose",
            ).value
        )
        self.controller_id = str(
            self.declare_parameter("controller_id", "FollowPath").value
        )
        self.goal_checker_id = str(
            self.declare_parameter(
                "goal_checker_id",
                "general_goal_checker",
            ).value
        )

        self.model_repo_path = str(
            self.declare_parameter("model_repo_path", "").value
        )
        self.model_config_path = str(
            self.declare_parameter("model_config_path", "").value
        )
        self.model_checkpoint_path = str(
            self.declare_parameter("model_checkpoint_path", "").value
        )
        self.map_yaml_path = str(
            self.declare_parameter("map_yaml_path", "").value
        )
        self.use_cuda = self._as_bool(self.declare_parameter("use_cuda", True).value)
        self.d_sample = int(self.declare_parameter("d_sample", 40).value)
        self.guidance_radius = float(
            self.declare_parameter("guidance_radius", 8.0).value
        )
        self.obs_len = int(self.declare_parameter("obs_len", 8).value)
        self.pred_len = int(self.declare_parameter("pred_len", 12).value)
        self.prediction_dt = float(
            self.declare_parameter("prediction_dt", 0.4).value
        )
        self.replan_period = float(
            self.declare_parameter("replan_period", 0.8).value
        )
        self.goal_tolerance = float(
            self.declare_parameter("goal_tolerance", 0.40).value
        )
        self.robot_radius = float(
            self.declare_parameter("robot_radius", 0.275).value
        )
        self.static_safety_margin = float(
            self.declare_parameter("static_safety_margin", 0.10).value
        )
        self.min_human_center_distance = float(
            self.declare_parameter("min_human_center_distance", 1.20).value
        )
        self.human_safety_margin = float(
            self.declare_parameter("human_safety_margin", 0.25).value
        )
        self.max_robot_speed = float(
            self.declare_parameter("max_robot_speed", 1.50).value
        )
        self.max_step_ratio = float(
            self.declare_parameter("max_step_ratio", 1.50).value
        )
        self.min_path_progress = float(
            self.declare_parameter("min_path_progress", 0.10).value
        )
        self.fallback_to_nav2 = self._as_bool(
            self.declare_parameter("fallback_to_nav2", True).value
        )

        self._robot: Optional[Agent] = None
        self._humans: Dict[int, Agent] = {}
        self._robot_history: deque[XY] = deque(maxlen=self.obs_len)
        self._human_histories: Dict[int, deque[XY]] = {}
        self._predicted_human_paths: Dict[int, List[XY]] = {}
        self._goal: Optional[PoseStamped] = None
        self._goal_generation = 0
        self._fallback_active = False
        self._last_follow_send = -math.inf
        self._follow_goal_handle = None
        self._navigate_goal_handle = None
        self._follow_goal_pending = False
        self._navigate_goal_pending = False
        self._navigate_dispatched_generation = -1
        self._last_status = ""
        self._runtime: Optional[GuidedSpubertRuntime] = None
        self._map_provider: Optional[OccupancyMapProvider] = None
        self._diagnostics_file = None

        if self.diagnostics_path:
            os.makedirs(os.path.dirname(self.diagnostics_path), exist_ok=True)
            self._diagnostics_file = open(
                self.diagnostics_path, "w", encoding="utf-8", buffering=1
            )
            self.get_logger().info(
                f"Writing guided runtime diagnostics to {self.diagnostics_path}"
            )

        self._robot_sub = self.create_subscription(
            Agent,
            self.robot_topic,
            self._on_robot,
            10,
        )
        self._humans_sub = self.create_subscription(
            Agents,
            self.humans_topic,
            self._on_humans,
            10,
        )
        self._goal_sub = self.create_subscription(
            PoseStamped,
            self.goal_topic,
            self._on_goal,
            10,
        )
        self._predicted_humans_sub = self.create_subscription(
            MarkerArray,
            self.predicted_humans_topic,
            self._on_predicted_humans,
            10,
        )
        self._path_pub = self.create_publisher(Path, self.path_topic, 10)
        self._marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 10)
        self._status_pub = self.create_publisher(String, self.status_topic, 10)
        self._follow_client = ActionClient(
            self,
            FollowPath,
            self.follow_path_action,
        )
        self._navigate_client = ActionClient(
            self,
            NavigateToPose,
            self.navigate_to_pose_action,
        )
        self._timer = self.create_timer(
            max(self.prediction_dt, 0.05),
            self._on_timer,
        )

        self._initialize_runtime()
        self._publish_status(
            f"ready mode={self.execution_mode} model_loaded={self._runtime is not None}"
        )
        self.get_logger().info(
            "SPU-BERT/Nav2 bridge ready: "
            f"mode={self.execution_mode}, goal={self.goal_topic}, path={self.path_topic}"
        )

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _initialize_runtime(self) -> None:
        if self.execution_mode == "nav2":
            return
        try:
            self._map_provider = OccupancyMapProvider(
                self.map_yaml_path,
                self.get_logger(),
            )
            self._runtime = GuidedSpubertRuntime(
                repo_path=self.model_repo_path,
                config_path=self.model_config_path,
                checkpoint_path=self.model_checkpoint_path,
                map_provider=self._map_provider,
                use_cuda=self.use_cuda,
                d_sample=self.d_sample,
                guidance_radius=self.guidance_radius,
                logger=self.get_logger(),
            )
            if self._runtime.obs_len != self.obs_len:
                raise RuntimeError(
                    f"obs_len mismatch: node={self.obs_len}, model={self._runtime.obs_len}"
                )
            if self._runtime.pred_len != self.pred_len:
                raise RuntimeError(
                    f"pred_len mismatch: node={self.pred_len}, model={self._runtime.pred_len}"
                )
        except Exception as exc:
            self._runtime = None
            self.get_logger().error(
                f"Guided robot model unavailable; Nav2 fallback remains active: {exc}"
            )

    def _on_robot(self, msg: Agent) -> None:
        self._robot = msg

    def _on_humans(self, msg: Agents) -> None:
        self._humans = {int(agent.id): agent for agent in msg.agents}

    def _on_goal(self, msg: PoseStamped) -> None:
        self._goal_generation += 1
        self._goal = msg
        self._fallback_active = False
        self._last_follow_send = -math.inf
        self._cancel_action(self._follow_goal_handle)
        self._cancel_action(self._navigate_goal_handle)
        self._follow_goal_handle = None
        self._navigate_goal_handle = None
        self._follow_goal_pending = False
        self._navigate_goal_pending = False
        self._navigate_dispatched_generation = -1
        self.get_logger().info(
            f"Received RViz final goal: ({msg.pose.position.x:.3f}, "
            f"{msg.pose.position.y:.3f})"
        )

        if self._runtime is None and self.execution_mode == "spubert":
            self._fallback_active = True
        # Humble bt_navigator already subscribes to /goal_pose. In standard
        # nav2/monitor mode it receives this same message directly; relaying it
        # again as NavigateToPose would preempt the goal with a duplicate.
        if self.execution_mode in {"nav2", "monitor"}:
            self._publish_status("goal_received_by_standard_nav2")
        elif self._runtime is None:
            self._activate_nav2("RViz goal relay")

    def _on_predicted_humans(self, msg: MarkerArray) -> None:
        predicted: Dict[int, List[XY]] = {}
        for marker in msg.markers:
            if marker.action == Marker.DELETEALL or "predicted_path" not in marker.ns:
                continue
            agent_id = self._agent_id_from_ns(marker.ns)
            if agent_id is None:
                continue
            points = marker.points[1:] if len(marker.points) > 1 else marker.points
            predicted[agent_id] = [
                (float(point.x), float(point.y)) for point in points
            ]
        self._predicted_human_paths = predicted

    def _on_timer(self) -> None:
        self._sample_histories()
        goal = self._goal
        robot = self._robot
        if goal is None or robot is None:
            return

        if self._distance_to_goal(robot, goal) <= self.goal_tolerance:
            self._cancel_active_navigation()
            self._publish_status("goal_reached")
            self._goal = None
            return

        if self.execution_mode == "nav2":
            return
        if self._fallback_active:
            if (
                self._navigate_dispatched_generation != self._goal_generation
                and self._navigate_goal_handle is None
                and not self._navigate_goal_pending
            ):
                self._activate_nav2("retrying Nav2 fallback")
            return
        if self._runtime is None:
            self._handle_invalid_path("model_unavailable")
            return

        try:
            result = self._runtime.predict(
                robot_history=list(self._robot_history),
                robot_yaw=float(robot.yaw),
                human_histories={
                    agent_id: list(history)
                    for agent_id, history in self._human_histories.items()
                },
                final_goal=(
                    float(goal.pose.position.x),
                    float(goal.pose.position.y),
                ),
            )
        except Exception as exc:
            self.get_logger().error(f"Guided robot inference failed: {exc}")
            self._record_event("inference_error", {"error": str(exc)})
            self._handle_invalid_path("inference_error")
            return

        diagnostics = self._path_diagnostics(result)
        valid, reason = self._validate_result(result, diagnostics)
        self._record_diagnostic(result, valid, reason, diagnostics)
        self._publish_debug(result, valid=valid, reason=reason)
        if not valid:
            self._handle_invalid_path(reason)
            return

        path_msg = self._path_message(result.path_world)
        self._path_pub.publish(path_msg)
        self._publish_status(
            f"guided_path_valid min_human={self._minimum_human_distance(result.path_world):.3f}"
        )
        if self.execution_mode == "monitor":
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._last_follow_send >= max(self.replan_period, self.prediction_dt):
            if self._send_follow_path(path_msg):
                self._last_follow_send = now
            else:
                self._handle_invalid_path("follow_path_server_unavailable")

    def _sample_histories(self) -> None:
        if self._robot is not None:
            self._robot_history.append(
                (
                    float(self._robot.position.position.x),
                    float(self._robot.position.position.y),
                )
            )
        active_ids = set(self._humans)
        for agent_id, agent in self._humans.items():
            history = self._human_histories.setdefault(
                agent_id,
                deque(maxlen=self.obs_len),
            )
            history.append(
                (
                    float(agent.position.position.x),
                    float(agent.position.position.y),
                )
            )
        for stale_id in set(self._human_histories) - active_ids:
            self._human_histories.pop(stale_id, None)

    def _validate_result(
        self,
        result: GuidedInferenceResult,
        diagnostics: Dict[str, Any],
    ) -> Tuple[bool, str]:
        if not result.selected_goal_valid:
            return False, "no_map_safe_goal"
        if not result.trajectory_map_safe or not result.execution_valid:
            return False, "model_map_check_failed"
        if len(result.path_world) != self.pred_len:
            return False, "invalid_path_length"
        if not all(
            math.isfinite(x) and math.isfinite(y) for x, y in result.path_world
        ):
            return False, "nonfinite_path"
        if self._map_provider is None:
            return False, "map_unavailable"

        if diagnostics["footprint_collision_count"] > 0:
            return False, "robot_footprint_collision"

        current = (
            float(self._robot.position.position.x),
            float(self._robot.position.position.y),
        )
        if diagnostics["max_step_m"] > diagnostics["allowed_step_m"]:
            return False, "kinematic_jump"

        if diagnostics["goal_progress_m"] < self.min_path_progress:
            return False, "insufficient_goal_progress"

        if diagnostics["min_human_clearance_m"] < 0.0:
            return False, "predicted_human_clearance"
        return True, "valid"

    def _path_diagnostics(self, result: GuidedInferenceResult) -> Dict[str, Any]:
        current = (
            float(self._robot.position.position.x),
            float(self._robot.position.position.y),
        )
        goal = (
            float(self._goal.pose.position.x),
            float(self._goal.pose.position.y),
        )
        footprint_radius = self.robot_radius + self.static_safety_margin
        collision_steps = [
            index
            for index, point in enumerate(result.path_world)
            if self._map_provider.path_collision_cost(
                [point], radius=footprint_radius, weight=1.0
            ) > 0.0
        ] if self._map_provider is not None else []

        previous = current
        step_distances = []
        for point in result.path_world:
            step_distances.append(math.hypot(point[0] - previous[0], point[1] - previous[1]))
            previous = point
        max_step_m = max(step_distances, default=0.0)
        max_step_index = step_distances.index(max_step_m) if step_distances else -1
        allowed_step_m = (
            max(self.max_robot_speed, 0.01)
            * max(self.prediction_dt, 0.01)
            * max(self.max_step_ratio, 1.0)
        )

        goal_dx = goal[0] - current[0]
        goal_dy = goal[1] - current[1]
        goal_distance = max(math.hypot(goal_dx, goal_dy), 1e-6)
        endpoint = result.path_world[-1] if result.path_world else current
        goal_progress_m = (
            (endpoint[0] - current[0]) * goal_dx
            + (endpoint[1] - current[1]) * goal_dy
        ) / goal_distance
        human = self._minimum_human_diagnostic(result.path_world)

        return {
            "selected_goal_valid": bool(result.selected_goal_valid),
            "trajectory_map_safe": bool(result.trajectory_map_safe),
            "execution_valid": bool(result.execution_valid),
            "footprint_radius_m": float(footprint_radius),
            "footprint_collision_steps": collision_steps,
            "footprint_collision_count": len(collision_steps),
            "step_distances_m": step_distances,
            "max_step_m": float(max_step_m),
            "max_step_index": int(max_step_index),
            "allowed_step_m": float(allowed_step_m),
            "goal_progress_m": float(goal_progress_m),
            "required_goal_progress_m": float(self.min_path_progress),
            **human,
        }

    def _minimum_human_diagnostic(self, path: Sequence[XY]) -> Dict[str, Any]:
        best = {
            "min_human_center_distance_m": math.inf,
            "min_human_required_distance_m": 0.0,
            "min_human_clearance_m": math.inf,
            "min_human_step": -1,
            "min_human_agent_id": -1,
            "min_human_position": None,
            "min_human_robot_position": None,
        }
        for step_index, (robot_x, robot_y) in enumerate(path):
            horizon = self.prediction_dt * (step_index + 1)
            for agent_id, human in self._humans.items():
                predicted = self._predicted_human_paths.get(agent_id, [])
                if predicted:
                    human_x, human_y = predicted[min(step_index, len(predicted) - 1)]
                else:
                    human_x = float(human.position.position.x) + float(human.velocity.linear.x) * horizon
                    human_y = float(human.position.position.y) + float(human.velocity.linear.y) * horizon
                center = math.hypot(robot_x - human_x, robot_y - human_y)
                required = max(
                    self.min_human_center_distance,
                    self.robot_radius + max(float(human.radius), 0.35) + self.human_safety_margin,
                )
                clearance = center - required
                if clearance < best["min_human_clearance_m"]:
                    best = {
                        "min_human_center_distance_m": float(center),
                        "min_human_required_distance_m": float(required),
                        "min_human_clearance_m": float(clearance),
                        "min_human_step": int(step_index),
                        "min_human_agent_id": int(agent_id),
                        "min_human_position": [float(human_x), float(human_y)],
                        "min_human_robot_position": [float(robot_x), float(robot_y)],
                    }
        return best

    def _record_diagnostic(
        self,
        result: GuidedInferenceResult,
        valid: bool,
        reason: str,
        diagnostics: Dict[str, Any],
    ) -> None:
        robot = self._robot
        goal = self._goal
        if self._diagnostics_file is None or robot is None or goal is None:
            return
        humans = []
        for agent_id, human in sorted(self._humans.items()):
            predicted = self._predicted_human_paths.get(agent_id, [])
            if not predicted:
                predicted = [
                    (
                        float(human.position.position.x)
                        + float(human.velocity.linear.x) * self.prediction_dt * step,
                        float(human.position.position.y)
                        + float(human.velocity.linear.y) * self.prediction_dt * step,
                    )
                    for step in range(1, self.pred_len + 1)
                ]
            humans.append({
                "id": int(agent_id),
                "current": [
                    float(human.position.position.x),
                    float(human.position.position.y),
                ],
                "radius": float(human.radius),
                "predicted": [
                    [float(x), float(y)]
                    for x, y in predicted
                ],
            })
        self._record_event("prediction", {
            "goal_generation": int(self._goal_generation),
            "valid": bool(valid),
            "reason": str(reason),
            "robot": [
                float(robot.position.position.x),
                float(robot.position.position.y),
            ],
            "robot_yaw": float(robot.yaw),
            "robot_history": [[float(x), float(y)] for x, y in self._robot_history],
            "final_goal": [
                float(goal.pose.position.x),
                float(goal.pose.position.y),
            ],
            "guidance_point": list(result.guidance_point_world),
            "candidate_goals": [list(point) for point in result.candidate_goals_world],
            "selected_goal": list(result.selected_goal_world),
            "path": [list(point) for point in result.path_world],
            "humans": humans,
            "metrics": diagnostics,
            "map_yaml_path": (
                str(self._map_provider.yaml_path)
                if self._map_provider is not None else ""
            ),
        })

    def _record_event(self, event: str, payload: Dict[str, Any]) -> None:
        if self._diagnostics_file is None:
            return
        record = {
            "event": event,
            "stamp_ns": int(self.get_clock().now().nanoseconds),
            "mode": self.execution_mode,
            **payload,
        }
        try:
            self._diagnostics_file.write(
                json.dumps(self._json_safe(record), ensure_ascii=False) + "\n"
            )
        except Exception as exc:
            self.get_logger().error(f"Disabling runtime diagnostics after write failure: {exc}")
            self._diagnostics_file.close()
            self._diagnostics_file = None

    @classmethod
    def _json_safe(cls, value):
        if hasattr(value, "item") and callable(value.item):
            return cls._json_safe(value.item())
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {str(key): cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_safe(item) for item in value]
        return value

    def _minimum_human_distance(
        self,
        path: Sequence[XY],
        *,
        clearance: bool = False,
    ) -> float:
        minimum = math.inf
        for step_index, (robot_x, robot_y) in enumerate(path):
            horizon = self.prediction_dt * (step_index + 1)
            for agent_id, human in self._humans.items():
                predicted = self._predicted_human_paths.get(agent_id, [])
                if predicted:
                    human_x, human_y = predicted[min(step_index, len(predicted) - 1)]
                else:
                    human_x = (
                        float(human.position.position.x)
                        + float(human.velocity.linear.x) * horizon
                    )
                    human_y = (
                        float(human.position.position.y)
                        + float(human.velocity.linear.y) * horizon
                    )
                center_distance = math.hypot(robot_x - human_x, robot_y - human_y)
                required = max(
                    self.min_human_center_distance,
                    self.robot_radius
                    + max(float(human.radius), 0.35)
                    + self.human_safety_margin,
                )
                value = center_distance - required if clearance else center_distance
                minimum = min(minimum, value)
        return minimum

    def _handle_invalid_path(self, reason: str) -> None:
        self._publish_status(f"guided_path_rejected reason={reason}")
        if self.execution_mode == "monitor":
            return
        self._cancel_action(self._follow_goal_handle)
        self._follow_goal_handle = None
        if self.fallback_to_nav2:
            self._fallback_active = True
            self._activate_nav2(f"fallback reason={reason}")

    def _activate_nav2(self, reason: str) -> bool:
        goal = self._goal
        if goal is None:
            return False
        if not self._navigate_client.wait_for_server(timeout_sec=0.05):
            self._publish_status(f"waiting_for_navigate_to_pose reason={reason}")
            return False
        if self._navigate_goal_pending:
            return True
        if self._navigate_dispatched_generation == self._goal_generation:
            return self._navigate_goal_handle is not None

        action_goal = NavigateToPose.Goal()
        action_goal.pose = goal
        generation = self._goal_generation
        self._navigate_goal_pending = True
        self._navigate_dispatched_generation = generation
        try:
            future = self._navigate_client.send_goal_async(action_goal)
        except Exception:
            self._navigate_goal_pending = False
            self._navigate_dispatched_generation = -1
            raise
        future.add_done_callback(
            lambda done, generation=generation: self._on_navigate_goal_response(
                done,
                generation,
            )
        )
        self._publish_status(f"nav2_goal_sent reason={reason}")
        return True

    def _on_navigate_goal_response(self, future, generation: int) -> None:
        self._navigate_goal_pending = False
        if generation != self._goal_generation:
            return
        try:
            handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"NavigateToPose request failed: {exc}")
            self._publish_status("nav2_goal_request_failed waiting_for_new_goal")
            self._goal = None
            return
        if not handle.accepted:
            self.get_logger().error("NavigateToPose goal was rejected")
            self._publish_status("nav2_goal_rejected waiting_for_new_goal")
            self._goal = None
            return
        self._navigate_goal_handle = handle
        handle.get_result_async().add_done_callback(
            lambda done, generation=generation: self._on_navigate_result(
                done,
                generation,
            )
        )

    def _on_navigate_result(self, future, generation: int) -> None:
        if generation != self._goal_generation:
            return
        self._navigate_goal_handle = None
        try:
            status = int(future.result().status)
        except Exception as exc:
            self.get_logger().error(f"NavigateToPose result failed: {exc}")
            self._goal = None
            return
        self._publish_status(f"nav2_result status={status}")
        self._goal = None

    def _send_follow_path(self, path_msg: Path) -> bool:
        if not self._follow_client.wait_for_server(timeout_sec=0.05):
            return False
        if self._follow_goal_pending:
            return True
        action_goal = FollowPath.Goal()
        action_goal.path = path_msg
        action_goal.controller_id = self.controller_id
        action_goal.goal_checker_id = self.goal_checker_id
        generation = self._goal_generation
        self._follow_goal_pending = True
        future = self._follow_client.send_goal_async(action_goal)
        future.add_done_callback(
            lambda done, generation=generation: self._on_follow_goal_response(
                done,
                generation,
            )
        )
        return True

    def _on_follow_goal_response(self, future, generation: int) -> None:
        self._follow_goal_pending = False
        if generation != self._goal_generation:
            return
        try:
            handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"FollowPath request failed: {exc}")
            self._handle_invalid_path("follow_path_request_failed")
            return
        if not handle.accepted:
            self.get_logger().error("FollowPath goal was rejected")
            self._handle_invalid_path("follow_path_rejected")
            return
        self._follow_goal_handle = handle

    def _path_message(self, path: Sequence[XY]) -> Path:
        message = Path()
        message.header.frame_id = "map"
        message.header.stamp = self.get_clock().now().to_msg()
        current = (
            float(self._robot.position.position.x),
            float(self._robot.position.position.y),
        )
        points = [current, *path]
        for index, (x, y) in enumerate(points):
            next_index = min(index + 1, len(points) - 1)
            prev_index = max(index - 1, 0)
            dx = points[next_index][0] - points[prev_index][0]
            dy = points[next_index][1] - points[prev_index][1]
            yaw = math.atan2(dy, dx) if math.hypot(dx, dy) > 1e-6 else float(self._robot.yaw)
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation = self._quaternion_from_yaw(yaw)
            message.poses.append(pose)
        return message

    def _publish_debug(
        self,
        result: GuidedInferenceResult,
        *,
        valid: bool,
        reason: str,
    ) -> None:
        robot = self._robot
        goal = self._goal
        if robot is None or goal is None:
            return
        stamp = self.get_clock().now().to_msg()
        z = max(float(robot.position.position.z), 0.05) + 0.35
        current = (
            float(robot.position.position.x),
            float(robot.position.position.y),
        )
        markers = MarkerArray()
        markers.markers.append(self._delete_all_marker(stamp))

        circle = Marker()
        circle.header.frame_id = "map"
        circle.header.stamp = stamp
        circle.ns = "robot_guidance_circle"
        circle.id = 1
        circle.type = Marker.LINE_STRIP
        circle.action = Marker.ADD
        circle.pose.orientation.w = 1.0
        circle.scale.x = 0.06
        circle.color = ColorRGBA(r=0.05, g=0.05, b=0.05, a=0.9)
        circle.lifetime = Duration(sec=2)
        for index in range(65):
            angle = 2.0 * math.pi * index / 64.0
            circle.points.append(
                Point(
                    x=current[0] + self.guidance_radius * math.cos(angle),
                    y=current[1] + self.guidance_radius * math.sin(angle),
                    z=z,
                )
            )

        goal_line = Marker()
        goal_line.header.frame_id = "map"
        goal_line.header.stamp = stamp
        goal_line.ns = "robot_goal_line"
        goal_line.id = 2
        goal_line.type = Marker.LINE_STRIP
        goal_line.action = Marker.ADD
        goal_line.pose.orientation.w = 1.0
        goal_line.scale.x = 0.035
        goal_line.color = ColorRGBA(r=0.10, g=0.85, b=0.30, a=0.85)
        goal_line.lifetime = Duration(sec=2)
        goal_line.points = [
            Point(x=current[0], y=current[1], z=z),
            Point(
                x=float(goal.pose.position.x),
                y=float(goal.pose.position.y),
                z=z,
            ),
        ]

        gp = self._point_marker(
            stamp=stamp,
            namespace="robot_guidance_point",
            marker_id=3,
            marker_type=Marker.CUBE,
            point=result.guidance_point_world,
            z=z,
            scale=0.25,
            color=ColorRGBA(r=1.0, g=0.80, b=0.0, a=1.0),
        )
        final_goal = self._point_marker(
            stamp=stamp,
            namespace="robot_final_goal",
            marker_id=4,
            marker_type=Marker.SPHERE,
            point=(
                float(goal.pose.position.x),
                float(goal.pose.position.y),
            ),
            z=z,
            scale=0.32,
            color=ColorRGBA(r=0.0, g=0.90, b=0.30, a=1.0),
        )

        candidates = Marker()
        candidates.header.frame_id = "map"
        candidates.header.stamp = stamp
        candidates.ns = "robot_mgp_candidate_goals"
        candidates.id = 5
        candidates.type = Marker.SPHERE_LIST
        candidates.action = Marker.ADD
        candidates.pose.orientation.w = 1.0
        candidates.scale.x = 0.13
        candidates.scale.y = 0.13
        candidates.scale.z = 0.13
        candidates.color = ColorRGBA(r=0.10, g=0.35, b=1.0, a=0.65)
        candidates.lifetime = Duration(sec=2)
        candidates.points = [
            Point(x=float(x), y=float(y), z=z)
            for x, y in result.candidate_goals_world
        ]

        selected = Marker()
        selected.header.frame_id = "map"
        selected.header.stamp = stamp
        selected.ns = "robot_spubert_selected_path"
        selected.id = 6
        selected.type = Marker.LINE_STRIP
        selected.action = Marker.ADD
        selected.pose.orientation.w = 1.0
        selected.scale.x = 0.09
        selected.color = (
            ColorRGBA(r=1.0, g=0.55, b=0.0, a=1.0)
            if valid
            else ColorRGBA(r=1.0, g=0.05, b=0.05, a=1.0)
        )
        selected.lifetime = Duration(sec=2)
        selected.points = [
            Point(x=current[0], y=current[1], z=z),
            *[
                Point(x=float(x), y=float(y), z=z)
                for x, y in result.path_world
            ],
        ]

        status = Marker()
        status.header.frame_id = "map"
        status.header.stamp = stamp
        status.ns = "robot_spubert_status"
        status.id = 7
        status.type = Marker.TEXT_VIEW_FACING
        status.action = Marker.ADD
        status.pose.position.x = current[0]
        status.pose.position.y = current[1]
        status.pose.position.z = z + 0.8
        status.pose.orientation.w = 1.0
        status.scale.z = 0.30
        status.color = (
            ColorRGBA(r=0.05, g=0.75, b=0.15, a=1.0)
            if valid
            else ColorRGBA(r=1.0, g=0.10, b=0.10, a=1.0)
        )
        status.text = "SPU-BERT valid" if valid else f"rejected: {reason}"
        status.lifetime = Duration(sec=2)

        markers.markers.extend(
            [circle, goal_line, gp, final_goal, candidates, selected, status]
        )
        self._marker_pub.publish(markers)

    def _point_marker(
        self,
        *,
        stamp,
        namespace: str,
        marker_id: int,
        marker_type: int,
        point: XY,
        z: float,
        scale: float,
        color: ColorRGBA,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.position.x = float(point[0])
        marker.pose.position.y = float(point[1])
        marker.pose.position.z = z
        marker.pose.orientation.w = 1.0
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        marker.color = color
        marker.lifetime = Duration(sec=2)
        return marker

    @staticmethod
    def _quaternion_from_yaw(yaw: float) -> Quaternion:
        half = 0.5 * float(yaw)
        return Quaternion(z=math.sin(half), w=math.cos(half))

    @staticmethod
    def _distance_to_goal(robot: Agent, goal: PoseStamped) -> float:
        return math.hypot(
            float(goal.pose.position.x) - float(robot.position.position.x),
            float(goal.pose.position.y) - float(robot.position.position.y),
        )

    @staticmethod
    def _agent_id_from_ns(namespace: str) -> Optional[int]:
        try:
            return int(namespace.split("_")[1])
        except (IndexError, ValueError):
            return None

    @staticmethod
    def _cancel_action(handle) -> None:
        if handle is None:
            return
        try:
            handle.cancel_goal_async()
        except Exception:
            pass

    def _cancel_active_navigation(self) -> None:
        self._cancel_action(self._follow_goal_handle)
        self._cancel_action(self._navigate_goal_handle)
        self._follow_goal_handle = None
        self._navigate_goal_handle = None
        self._follow_goal_pending = False
        self._navigate_goal_pending = False

    @staticmethod
    def _delete_all_marker(stamp) -> Marker:
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = stamp
        marker.action = Marker.DELETEALL
        return marker

    def _publish_status(self, message: str) -> None:
        if message == self._last_status:
            return
        self._last_status = message
        self._status_pub.publish(String(data=message))
        self.get_logger().info(message)

    def destroy_node(self):
        self._cancel_active_navigation()
        if self._diagnostics_file is not None:
            self._diagnostics_file.close()
            self._diagnostics_file = None
        return super().destroy_node()


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = SpubertNav2BridgeNode()
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
