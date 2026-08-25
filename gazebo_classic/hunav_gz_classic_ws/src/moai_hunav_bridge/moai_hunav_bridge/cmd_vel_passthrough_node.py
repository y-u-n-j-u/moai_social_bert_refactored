#!/usr/bin/env python3
from __future__ import annotations

from math import cos, hypot, sin

import rclpy
from geometry_msgs.msg import Twist
from hunav_msgs.msg import Agent, Agents
from nav_msgs.msg import Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool

from .continuous_avoidance_teacher import (
    AvoidanceCommand,
    MovingHuman,
    choose_avoidance_side,
    filter_suppressed_humans,
    lane_tracking_command,
    path_frame_from_points,
    route_outward_side,
    select_continuous_avoidance_command,
    temporary_lane_is_clear,
)
from .human_yield_safety import (
    filtered_velocity_from_positions,
    is_separating_from_stationary_robot,
    minimum_path_predicted_clearance,
    minimum_predicted_clearance,
)


class CmdVelPassthroughNode(Node):
    """Forward Nav2 commands when the PMB2 velocity smoother drops output."""

    def __init__(self) -> None:
        super().__init__("cmd_vel_passthrough")
        input_topic = str(self.declare_parameter("input_topic", "/cmd_vel_nav").value)
        output_topic = str(self.declare_parameter("output_topic", "/cmd_vel").value)
        avoidance_status_topic = str(
            self.declare_parameter(
                "avoidance_status_topic",
                "/moai/human_avoidance_active",
            ).value
        )
        legacy_human_yield_enabled = bool(
            self.declare_parameter("human_yield_enabled", False).value
        )
        requested_avoidance_mode = str(
            self.declare_parameter("human_avoidance_mode", "").value
        ).strip().lower()
        if not requested_avoidance_mode:
            requested_avoidance_mode = (
                "yield" if legacy_human_yield_enabled else "off"
            )
        if requested_avoidance_mode not in {"off", "yield", "continuous"}:
            self.get_logger().warning(
                f"Unknown human_avoidance_mode={requested_avoidance_mode!r}; using off"
            )
            requested_avoidance_mode = "off"
        self.human_avoidance_mode = requested_avoidance_mode
        self.human_yield_enabled = self.human_avoidance_mode == "yield"
        robot_topic = str(self.declare_parameter("robot_topic", "/robot_states").value)
        human_states_topic = str(
            self.declare_parameter("human_states_topic", "/human_states").value
        )
        global_path_topic = str(
            self.declare_parameter("global_path_topic", "/plan").value
        )
        self.safety_distance = float(
            self.declare_parameter("human_yield_safety_distance", 1.45).value
        )
        self.release_distance = float(
            self.declare_parameter("human_yield_release_distance", 1.60).value
        )
        self.prediction_horizon = float(
            self.declare_parameter("human_yield_prediction_horizon", 6.0).value
        )
        self.prediction_step = float(
            self.declare_parameter("human_yield_prediction_step", 0.10).value
        )
        self.path_speed_floor = float(
            self.declare_parameter("human_yield_path_speed_floor", 0.90).value
        )
        self.crossing_horizon = float(
            self.declare_parameter("human_yield_crossing_horizon", 15.0).value
        )
        self.stopping_buffer = float(
            self.declare_parameter("human_yield_stopping_buffer", 0.60).value
        )
        self.state_timeout = float(
            self.declare_parameter("human_yield_state_timeout", 0.50).value
        )
        self.velocity_smoothing_alpha = float(
            self.declare_parameter("human_yield_velocity_smoothing_alpha", 0.50).value
        )
        self.maximum_human_speed = float(
            self.declare_parameter("human_yield_maximum_human_speed", 2.0).value
        )
        self.continuous_safety_distance = float(
            self.declare_parameter(
                "continuous_avoidance_safety_distance", 1.20
            ).value
        )
        self.continuous_trigger_distance = float(
            self.declare_parameter(
                "continuous_avoidance_trigger_distance", 1.60
            ).value
        )
        self.continuous_horizon = float(
            self.declare_parameter("continuous_avoidance_horizon", 3.5).value
        )
        self.continuous_static_route_horizon = float(
            self.declare_parameter(
                "continuous_avoidance_static_route_horizon", 5.0
            ).value
        )
        self.continuous_step = float(
            self.declare_parameter("continuous_avoidance_step", 0.10).value
        )
        self.continuous_activation_distance = float(
            self.declare_parameter(
                "continuous_avoidance_activation_distance", 6.8
            ).value
        )
        self.continuous_preferred_speed = float(
            self.declare_parameter(
                "continuous_avoidance_preferred_speed", 0.70
            ).value
        )
        self.continuous_minimum_forward_speed = float(
            self.declare_parameter(
                "continuous_avoidance_minimum_forward_speed", 0.35
            ).value
        )
        self.continuous_maximum_forward_speed = float(
            self.declare_parameter(
                "continuous_avoidance_maximum_forward_speed", 0.80
            ).value
        )
        self.continuous_maximum_angular_speed = float(
            self.declare_parameter(
                "continuous_avoidance_maximum_angular_speed", 1.0
            ).value
        )
        self.continuous_steering_duration = float(
            self.declare_parameter(
                "continuous_avoidance_steering_duration", 1.0
            ).value
        )
        self.continuous_lateral_offset = float(
            self.declare_parameter(
                "continuous_avoidance_lateral_offset", 1.60
            ).value
        )
        self.continuous_static_route_lateral_offset = float(
            self.declare_parameter(
                "continuous_avoidance_static_route_lateral_offset", 2.00
            ).value
        )
        self.continuous_lane_lookahead = float(
            self.declare_parameter(
                "continuous_avoidance_lane_lookahead", 2.00
            ).value
        )
        self.continuous_static_route_lane_lookahead = float(
            self.declare_parameter(
                "continuous_avoidance_static_route_lane_lookahead", 1.20
            ).value
        )
        self.continuous_heading_gain = float(
            self.declare_parameter(
                "continuous_avoidance_heading_gain", 1.50
            ).value
        )
        self.continuous_release_lateral = float(
            self.declare_parameter(
                "continuous_avoidance_release_lateral", 0.80
            ).value
        )
        self.continuous_center_tolerance = float(
            self.declare_parameter(
                "continuous_avoidance_center_tolerance", 0.30
            ).value
        )
        self.continuous_recovery_lookahead = float(
            self.declare_parameter(
                "continuous_avoidance_recovery_lookahead", 1.50
            ).value
        )

        self._robot: Agent | None = None
        self._humans: Agents | None = None
        self._robot_at: float | None = None
        self._humans_at: float | None = None
        self._global_path: list[tuple[float, float]] = []
        self._human_positions: dict[int, tuple[float, float]] = {}
        self._human_velocities: dict[int, tuple[float, float]] = {}
        self._yielding = False
        self._yield_agent_id: int | None = None
        self._continuous_avoiding = False
        self._continuous_agent_id: int | None = None
        self._continuous_avoidance_side = 0
        self._continuous_started_at: float | None = None
        self._continuous_reference_path: list[tuple[float, float]] = []
        self._continuous_reference_goal: tuple[float, float] | None = None
        self._continuous_recovering = False
        self._continuous_recovery_started_at: float | None = None
        self._continuous_recovery_agent_id: int | None = None
        self._continuous_recovery_side = 0
        self._continuous_suppressed_agent_ids: set[int] = set()
        self._stale_warning_emitted = False
        self._publisher = self.create_publisher(Twist, output_topic, 10)
        self._avoidance_status_publisher = self.create_publisher(
            Bool,
            avoidance_status_topic,
            10,
        )
        self._publish_avoidance_status(False)
        self._subscription = self.create_subscription(
            Twist, input_topic, self._on_command, 10
        )
        if self.human_avoidance_mode != "off":
            self._robot_subscription = self.create_subscription(
                Agent, robot_topic, self._on_robot, 10
            )
            self._human_subscription = self.create_subscription(
                Agents, human_states_topic, self._on_humans, 10
            )
            self._path_subscription = self.create_subscription(
                Path, global_path_topic, self._on_path, 10
            )
        self.get_logger().info(
            f"Forwarding PMB2 Nav2 velocity commands: {input_topic} -> {output_topic}; "
            f"human_avoidance_mode={self.human_avoidance_mode}, "
            f"distance={self.safety_distance:.2f} m, horizon={self.prediction_horizon:.1f} s, "
            f"crossing_horizon={self.crossing_horizon:.1f} s"
        )

    def _on_robot(self, msg: Agent) -> None:
        self._robot = msg
        self._robot_at = self._now()

    def _on_humans(self, msg: Agents) -> None:
        now = self._now()
        previous_time = self._humans_at
        elapsed = now - previous_time if previous_time is not None else 0.0
        new_positions: dict[int, tuple[float, float]] = {}
        new_velocities: dict[int, tuple[float, float]] = {}
        for human in msg.agents:
            human_id = int(human.id)
            position = (
                float(human.position.position.x),
                float(human.position.position.y),
            )
            message_velocity = (
                float(human.velocity.linear.x),
                float(human.velocity.linear.y),
            )
            previous_position = self._human_positions.get(human_id)
            if previous_position is None or elapsed <= 1e-6 or elapsed > self.state_timeout:
                velocity = message_velocity
            else:
                velocity = filtered_velocity_from_positions(
                    previous_position=previous_position,
                    current_position=position,
                    elapsed=elapsed,
                    previous_velocity=self._human_velocities.get(human_id, message_velocity),
                    smoothing_alpha=self.velocity_smoothing_alpha,
                    maximum_speed=self.maximum_human_speed,
                )
            new_positions[human_id] = position
            new_velocities[human_id] = velocity

        self._human_positions = new_positions
        self._human_velocities = new_velocities
        self._humans = msg
        self._humans_at = now

    def _on_path(self, msg: Path) -> None:
        new_path = [
            (float(pose.pose.position.x), float(pose.pose.position.y))
            for pose in msg.poses
        ]
        self._global_path = new_path
        if self.human_avoidance_mode != "continuous" or len(new_path) < 2:
            return

        new_goal = new_path[-1]
        goal_changed = (
            self._continuous_reference_goal is None
            or hypot(
                new_goal[0] - self._continuous_reference_goal[0],
                new_goal[1] - self._continuous_reference_goal[1],
            )
            >= 0.75
        )
        if not self._continuous_reference_path or goal_changed:
            # Freeze the first route for this goal. Later Nav2 replans can bend
            # around a person or the robot's current offset; using them again
            # would stack another avoidance offset on top of the first one.
            self._continuous_reference_path = list(new_path)
            self._continuous_reference_goal = new_goal
            self._continuous_recovery_agent_id = None
            self._continuous_recovery_side = 0
            self._continuous_suppressed_agent_ids.clear()

    def _on_command(self, msg: Twist) -> None:
        if self.human_avoidance_mode == "off":
            self._publisher.publish(msg)
            return

        if not self._states_are_fresh():
            if not self._stale_warning_emitted:
                stale_action = (
                    "holding zero velocity"
                    if self.human_yield_enabled
                    else "passing the nominal command"
                )
                self.get_logger().warning(
                    "Human avoidance states are stale; " + stale_action
                )
                self._stale_warning_emitted = True
            self._publisher.publish(Twist() if self.human_yield_enabled else msg)
            return
        self._stale_warning_emitted = False

        if self.human_avoidance_mode == "continuous":
            self._publish_continuous_command(msg)
            return

        if self._yielding:
            if self._can_release(msg):
                self._yielding = False
                self._yield_agent_id = None
                self._publish_avoidance_status(False)
                self.get_logger().info("human_yield_released")
            else:
                self._publisher.publish(Twist())
                return

        conflict = self._first_conflict(msg)
        if conflict is not None:
            human_id, clearance, time_to_closest, source = conflict
            estimated_velocity = self._human_velocities.get(human_id, (0.0, 0.0))
            self._yielding = True
            self._yield_agent_id = human_id
            self._publish_avoidance_status(True)
            self.get_logger().info(
                "human_yield_started: "
                f"agent={human_id}, predicted_clearance={clearance:.2f} m, "
                f"ttc={time_to_closest:.2f} s, source={source}, "
                f"human_velocity=({estimated_velocity[0]:.2f},{estimated_velocity[1]:.2f}) m/s"
            )
            self._publisher.publish(Twist())
            return

        self._publisher.publish(msg)

    def _publish_continuous_command(self, nominal_command: Twist) -> None:
        assert self._robot is not None
        assert self._humans is not None
        robot_position = (
            float(self._robot.position.position.x),
            float(self._robot.position.position.y),
        )
        all_humans = [
            MovingHuman(
                agent_id=int(human.id),
                x=float(human.position.position.x),
                y=float(human.position.position.y),
                vx=self._velocity_for_human(human)[0],
                vy=self._velocity_for_human(human)[1],
            )
            for human in self._humans.agents
        ]
        humans, self._continuous_suppressed_agent_ids = filter_suppressed_humans(
            robot_position=robot_position,
            humans=all_humans,
            suppressed_agent_ids=self._continuous_suppressed_agent_ids,
            rearm_distance=self.continuous_activation_distance,
        )
        current_path_direction, current_path_reference = self._path_frame(
            robot_position
        )
        # The route is frozen per goal, but its local tangent must advance with
        # the robot. Holding the tangent from avoidance start makes a curved
        # bypass appear straight and can steer back toward the pedestrian.
        path_direction = current_path_direction
        path_reference = current_path_reference
        direction_length = max(hypot(*path_direction), 1e-6)
        direction = (
            path_direction[0] / direction_length,
            path_direction[1] / direction_length,
        )
        normal = (-direction[1], direction[0])
        current_lateral_offset = (
            (robot_position[0] - path_reference[0]) * normal[0]
            + (robot_position[1] - path_reference[1]) * normal[1]
        )
        static_route_side = route_outward_side(self._continuous_reference_path)
        prediction_horizon = (
            self.continuous_static_route_horizon
            if static_route_side != 0
            else self.continuous_horizon
        )
        probe = select_continuous_avoidance_command(
            robot_position=robot_position,
            robot_yaw=float(self._robot.yaw),
            nominal_linear_speed=float(nominal_command.linear.x),
            nominal_angular_speed=float(nominal_command.angular.z),
            humans=humans,
            path_direction=path_direction,
            safety_distance=self.continuous_safety_distance,
            trigger_distance=self.continuous_trigger_distance,
            horizon=prediction_horizon,
            step=self.continuous_step,
            activation_distance=self.continuous_activation_distance,
            preferred_speed=self.continuous_preferred_speed,
            minimum_forward_speed=self.continuous_minimum_forward_speed,
            maximum_forward_speed=self.continuous_maximum_forward_speed,
            maximum_angular_speed=self.continuous_maximum_angular_speed,
            steering_duration=self.continuous_steering_duration,
        )
        if static_route_side != 0:
            path_probe = self._static_route_conflict_probe(
                robot_position=robot_position,
                nominal_command=nominal_command,
                humans=humans,
            )
            if path_probe is not None and (
                not probe.intervention
                or path_probe.predicted_clearance < probe.predicted_clearance
            ):
                probe = path_probe
        humans_by_id = {human.agent_id: human for human in humans}
        recovery_resume_side = 0
        if self._continuous_recovering:
            if abs(current_lateral_offset) <= self.continuous_center_tolerance:
                self._complete_continuous_recovery(current_lateral_offset)
                self._publisher.publish(nominal_command)
                return

            recovery_interrupted = (
                probe.intervention and probe.agent_id in humans_by_id
            )
            if not recovery_interrupted:
                linear_speed, angular_speed = lane_tracking_command(
                    robot_yaw=float(self._robot.yaw),
                    path_direction=path_direction,
                    current_lateral_offset=current_lateral_offset,
                    avoidance_side=1,
                    lateral_offset=0.0,
                    lane_lookahead=self.continuous_recovery_lookahead,
                    preferred_speed=self.continuous_preferred_speed,
                    minimum_forward_speed=self.continuous_minimum_forward_speed,
                    maximum_forward_speed=self.continuous_maximum_forward_speed,
                    maximum_angular_speed=self.continuous_maximum_angular_speed,
                    heading_gain=self.continuous_heading_gain,
                )
                self._publish_continuous_twist(
                    nominal_command,
                    linear_speed,
                    angular_speed,
                )
                return

            self.get_logger().warning(
                "continuous_human_avoidance_recovery_interrupted: "
                f"agent={probe.agent_id}, "
                f"lateral_offset={current_lateral_offset:.2f} m"
            )
            recovery_resume_side = self._continuous_recovery_side
            self._continuous_recovering = False
            self._continuous_recovery_started_at = None
            self._continuous_recovery_agent_id = None
            self._continuous_recovery_side = 0

        just_started = False
        if (
            not self._continuous_avoiding
            and probe.intervention
            and probe.agent_id in humans_by_id
        ):
            tracked_human = humans_by_id[probe.agent_id]
            self._continuous_avoiding = True
            self._continuous_agent_id = probe.agent_id
            human_avoidance_side = choose_avoidance_side(
                path_direction=path_direction,
                robot_yaw=float(self._robot.yaw),
                human_position=(tracked_human.x, tracked_human.y),
                human_velocity=(tracked_human.vx, tracked_human.vy),
                path_reference=path_reference,
            )
            self._continuous_avoidance_side = recovery_resume_side or (
                static_route_side
                if static_route_side != 0
                else human_avoidance_side
            )
            self._continuous_started_at = self._now()
            self._publish_avoidance_status(True)
            just_started = True

        tracked_human = humans_by_id.get(self._continuous_agent_id)
        if self._continuous_avoiding and tracked_human is None:
            self._release_continuous_avoidance(
                "tracked human state disappeared",
                current_lateral_offset,
            )
            self._publisher.publish(nominal_command)
            return

        if not self._continuous_avoiding or tracked_human is None:
            self._publisher.publish(nominal_command)
            return

        elapsed = (
            self._now() - self._continuous_started_at
            if self._continuous_started_at is not None
            else 0.0
        )
        if (
            not just_started
            and elapsed >= 1.0
            and temporary_lane_is_clear(
                robot_position=robot_position,
                human=tracked_human,
                path_direction=path_direction,
                avoidance_side=self._continuous_avoidance_side,
                release_lateral=self.continuous_release_lateral,
                release_distance=self.release_distance,
            )
        ):
            self._release_continuous_avoidance(
                "human cleared the temporary lane",
                current_lateral_offset,
            )
            self._publisher.publish(nominal_command)
            return

        linear_speed, angular_speed = lane_tracking_command(
            robot_yaw=float(self._robot.yaw),
            path_direction=path_direction,
            current_lateral_offset=current_lateral_offset,
            avoidance_side=self._continuous_avoidance_side,
            lateral_offset=(
                self.continuous_static_route_lateral_offset
                if static_route_side != 0
                else self.continuous_lateral_offset
            ),
            lane_lookahead=(
                self.continuous_static_route_lane_lookahead
                if static_route_side != 0
                else self.continuous_lane_lookahead
            ),
            preferred_speed=self.continuous_preferred_speed,
            minimum_forward_speed=self.continuous_minimum_forward_speed,
            maximum_forward_speed=self.continuous_maximum_forward_speed,
            maximum_angular_speed=self.continuous_maximum_angular_speed,
            heading_gain=self.continuous_heading_gain,
        )

        if just_started:
            self.get_logger().info(
                "continuous_human_avoidance_started: "
                f"agent={self._continuous_agent_id}, "
                f"side={self._continuous_avoidance_side:+d}, "
                f"route_bias={static_route_side:+d}, "
                f"predicted_clearance={probe.predicted_clearance:.2f} m, "
                f"human_velocity=({tracked_human.vx:.2f},"
                f"{tracked_human.vy:.2f}) m/s, "
                f"command=({linear_speed:.2f} m/s, {angular_speed:.2f} rad/s)"
            )

        self._publish_continuous_twist(
            nominal_command,
            linear_speed,
            angular_speed,
        )

    def _publish_continuous_twist(
        self,
        nominal_command: Twist,
        linear_speed: float,
        angular_speed: float,
    ) -> None:
        output = Twist()
        output.linear.x = linear_speed
        output.linear.y = nominal_command.linear.y
        output.linear.z = nominal_command.linear.z
        output.angular.x = nominal_command.angular.x
        output.angular.y = nominal_command.angular.y
        output.angular.z = angular_speed
        self._publisher.publish(output)

    def _static_route_conflict_probe(
        self,
        *,
        robot_position: tuple[float, float],
        nominal_command: Twist,
        humans: tuple[MovingHuman, ...],
    ) -> AvoidanceCommand | None:
        best_clearance = float("inf")
        best_agent_id: int | None = None
        path_speed = max(
            abs(float(nominal_command.linear.x)),
            self.continuous_preferred_speed,
        )
        for human in humans:
            if hypot(
                human.x - robot_position[0],
                human.y - robot_position[1],
            ) > self.continuous_activation_distance:
                continue
            clearance, _ = minimum_path_predicted_clearance(
                path_points=self._continuous_reference_path,
                robot_position=robot_position,
                path_speed=path_speed,
                human_position=(human.x, human.y),
                human_velocity=(human.vx, human.vy),
                horizon=self.continuous_static_route_horizon,
                step=self.continuous_step,
            )
            if clearance < best_clearance:
                best_clearance = clearance
                best_agent_id = human.agent_id

        if (
            best_agent_id is None
            or best_clearance >= self.continuous_trigger_distance
        ):
            return None
        return AvoidanceCommand(
            linear_speed=float(nominal_command.linear.x),
            angular_speed=float(nominal_command.angular.z),
            intervention=True,
            predicted_clearance=best_clearance,
            agent_id=best_agent_id,
        )

    def _release_continuous_avoidance(
        self,
        reason: str,
        current_lateral_offset: float,
    ) -> None:
        self.get_logger().info(
            f"continuous_human_avoidance_released: reason={reason}"
        )
        self._continuous_recovery_agent_id = self._continuous_agent_id
        self._continuous_recovery_side = self._continuous_avoidance_side
        self._continuous_avoiding = False
        self._continuous_agent_id = None
        self._continuous_avoidance_side = 0
        self._continuous_started_at = None
        self._continuous_recovering = True
        self._continuous_recovery_started_at = self._now()
        self._publish_avoidance_status(True)
        self.get_logger().info(
            "continuous_human_avoidance_recovery_started: "
            f"lateral_offset={current_lateral_offset:.2f} m"
        )

    def _complete_continuous_recovery(
        self,
        current_lateral_offset: float,
    ) -> None:
        elapsed = (
            self._now() - self._continuous_recovery_started_at
            if self._continuous_recovery_started_at is not None
            else 0.0
        )
        self._continuous_recovering = False
        self._continuous_recovery_started_at = None
        if self._continuous_recovery_agent_id is not None:
            self._continuous_suppressed_agent_ids.add(
                self._continuous_recovery_agent_id
            )
        self._continuous_recovery_agent_id = None
        self._continuous_recovery_side = 0
        self._publish_avoidance_status(False)
        self.get_logger().info(
            "continuous_human_avoidance_recovery_completed: "
            f"lateral_offset={current_lateral_offset:.2f} m, "
            f"elapsed={elapsed:.2f} s"
        )

    def _publish_avoidance_status(self, active: bool) -> None:
        message = Bool()
        message.data = bool(active)
        self._avoidance_status_publisher.publish(message)

    def _path_frame(
        self, robot_position: tuple[float, float]
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        assert self._robot is not None
        reference_path = self._continuous_reference_path or self._global_path
        return path_frame_from_points(
            path_points=reference_path,
            robot_position=robot_position,
            fallback_yaw=float(self._robot.yaw),
        )

    def _first_conflict(self, command: Twist) -> tuple[int, float, float, str] | None:
        assert self._robot is not None
        assert self._humans is not None
        robot_position = (
            float(self._robot.position.position.x),
            float(self._robot.position.position.y),
        )
        for human in self._humans.agents:
            human_position = (
                float(human.position.position.x),
                float(human.position.position.y),
            )
            human_velocity = self._velocity_for_human(human)
            clearance, time_to_closest = minimum_predicted_clearance(
                robot_position=robot_position,
                robot_yaw=float(self._robot.yaw),
                linear_speed=float(command.linear.x),
                angular_speed=float(command.angular.z),
                human_position=human_position,
                human_velocity=human_velocity,
                horizon=self.prediction_horizon,
                step=self.prediction_step,
            )
            source = "command"
            path_clearance, path_time_to_closest = minimum_path_predicted_clearance(
                path_points=self._global_path,
                robot_position=robot_position,
                path_speed=max(abs(float(command.linear.x)), self.path_speed_floor),
                human_position=human_position,
                human_velocity=human_velocity,
                horizon=self.prediction_horizon,
                step=self.prediction_step,
            )
            if path_clearance < clearance:
                clearance = path_clearance
                time_to_closest = path_time_to_closest
                source = "global_path"
            stationary_clearance, stationary_time_to_closest = minimum_predicted_clearance(
                robot_position=robot_position,
                robot_yaw=float(self._robot.yaw),
                linear_speed=0.0,
                angular_speed=0.0,
                human_position=human_position,
                human_velocity=human_velocity,
                horizon=self.crossing_horizon,
                step=self.prediction_step,
            )
            if stationary_clearance < self.safety_distance + self.stopping_buffer:
                return (
                    int(human.id),
                    stationary_clearance,
                    stationary_time_to_closest,
                    "reserved_crossing",
                )
            if clearance < self.safety_distance:
                return int(human.id), clearance, time_to_closest, source
        return None

    def _can_release(self, command: Twist) -> bool:
        assert self._robot is not None
        assert self._humans is not None
        if self._first_conflict(command) is not None:
            return False
        robot_position = (
            float(self._robot.position.position.x),
            float(self._robot.position.position.y),
        )
        for human in self._humans.agents:
            human_position = (
                float(human.position.position.x),
                float(human.position.position.y),
            )
            human_velocity = self._velocity_for_human(human)
            current_distance, _ = minimum_predicted_clearance(
                robot_position=robot_position,
                robot_yaw=float(self._robot.yaw),
                linear_speed=0.0,
                angular_speed=0.0,
                human_position=human_position,
                human_velocity=human_velocity,
                horizon=0.0,
                step=self.prediction_step,
            )
            future_clearance, _ = minimum_predicted_clearance(
                robot_position=robot_position,
                robot_yaw=float(self._robot.yaw),
                linear_speed=0.0,
                angular_speed=0.0,
                human_position=human_position,
                human_velocity=human_velocity,
                horizon=self.prediction_horizon,
                step=self.prediction_step,
            )
            if (
                current_distance < self.release_distance
                or future_clearance < self.safety_distance
            ):
                return False
            if (
                int(human.id) == self._yield_agent_id
                and not is_separating_from_stationary_robot(
                    robot_position,
                    human_position,
                    human_velocity,
                )
            ):
                return False
        return True

    def _velocity_for_human(self, human: Agent) -> tuple[float, float]:
        return self._human_velocities.get(
            int(human.id),
            (
                float(human.velocity.linear.x),
                float(human.velocity.linear.y),
            ),
        )

    def _states_are_fresh(self) -> bool:
        if self._robot is None or self._humans is None:
            return False
        now = self._now()
        return self._is_fresh(self._robot_at, now) and self._is_fresh(self._humans_at, now)

    def _is_fresh(self, stamp: float | None, now: float) -> bool:
        if stamp is None:
            return False
        if self.state_timeout <= 0.0 or now < stamp:
            return True
        return now - stamp <= self.state_timeout

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CmdVelPassthroughNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
