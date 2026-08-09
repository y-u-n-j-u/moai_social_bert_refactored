#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import pickle
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from hunav_msgs.msg import Agent, Agents
from nav_msgs.msg import Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from .guided_spubert_runtime import guidance_point_along_path


class JackalTeleopDatasetLoggerNode(Node):
    """Record robot trajectories in MOAI all_trajs format.

    Row 0 is always the robot target. Rows 1: are humans used as neighbor
    context. Human future positions are recorded for analysis, while model
    inputs still use only the observed neighbor history.
    """

    def __init__(self) -> None:
        super().__init__("robot_target_dataset_logger")

        self.enabled = self._as_bool(self.declare_parameter("enabled", False).value)
        self.robot_topic = str(self.declare_parameter("robot_topic", "/robot_states").value)
        self.human_states_topic = str(self.declare_parameter("human_states_topic", "/human_states").value)
        self.goal_topic = str(self.declare_parameter("goal_topic", "/goal_pose").value)
        self.global_path_topic = str(
            self.declare_parameter("global_path_topic", "/plan").value
        )
        self.scenario_name = str(
            self.declare_parameter("scenario_name", "unknown").value
        )
        self.map_yaml_path = str(
            self.declare_parameter("map_yaml_path", "").value
        )
        self.robot_path_planner = str(
            self.declare_parameter("robot_path_planner", "unknown").value
        )
        self.agent_motion_model = str(
            self.declare_parameter("agent_motion_model", "unknown").value
        )
        self.pedestrians_avoid_robot = self._as_bool(
            self.declare_parameter("pedestrians_avoid_robot", True).value
        )
        self.agents_wait_for_goal = self._as_bool(
            self.declare_parameter("agents_wait_for_goal", False).value
        )
        self.collection_seed = int(
            self.declare_parameter("collection_seed", -1).value
        )
        self.output_path = str(
            self.declare_parameter(
                "output_path",
                "/home/hunav_gz_classic_ws/moai_recordings/robot_target_all_trajs.pkl",
            ).value
        )
        self.obs_len = int(self.declare_parameter("obs_len", 8).value)
        self.pred_len = int(self.declare_parameter("pred_len", 12).value)
        self.record_dt = float(self.declare_parameter("record_dt", 0.4).value)
        self.guidance_point_radius = float(self.declare_parameter("guidance_point_radius", 8.0).value)
        self.global_path_goal_tolerance = float(
            self.declare_parameter("global_path_goal_tolerance", 1.0).value
        )
        self.goal_reached_tolerance = float(self.declare_parameter("goal_reached_tolerance", 0.6).value)
        self.episode_timeout = float(self.declare_parameter("episode_timeout", 60.0).value)
        self.require_goal = self._as_bool(self.declare_parameter("require_goal", True).value)
        self.sample_stride = int(self.declare_parameter("sample_stride", 4).value)
        self.flush_every = int(self.declare_parameter("flush_every", 10).value)
        self.max_samples = int(self.declare_parameter("max_samples", 0).value)
        self.stale_timeout = float(self.declare_parameter("stale_timeout", 0.5).value)
        self.timer_rate = float(self.declare_parameter("timer_rate", 20.0).value)

        self.seq_len = self.obs_len + self.pred_len
        self._latest_robot: Optional[Tuple[float, float]] = None
        self._latest_humans: Dict[int, Tuple[float, float]] = {}
        self._latest_goal: Optional[Tuple[float, float]] = None
        self._latest_global_path: List[Tuple[float, float]] = []
        self._goal_active = False
        self._has_received_goal = False
        self._last_robot_stamp: Optional[float] = None
        self._last_humans_stamp: Optional[float] = None
        self._last_goal_stamp: Optional[float] = None
        self._last_global_path_stamp: Optional[float] = None
        self._last_record_stamp: Optional[float] = None
        self._frame_count = 0
        self._episode_id = 0
        self._last_flushed_sample_count = 0
        self._frames: Deque[Dict[str, Any]] = deque(maxlen=self.seq_len)
        # Keep samples pending until the robot actually reaches the episode
        # goal. Timed-out, aborted, and interrupted runs must not contaminate
        # the training file with stationary tails.
        self._pending_samples: List[np.ndarray] = []
        self._pending_meta: List[Dict[str, Any]] = []
        self._samples: List[np.ndarray] = []
        self._sample_meta: List[Dict[str, Any]] = []

        self._robot_sub = self.create_subscription(Agent, self.robot_topic, self._on_robot, 10)
        self._humans_sub = self.create_subscription(Agents, self.human_states_topic, self._on_humans, 10)
        self._goal_sub = self.create_subscription(PoseStamped, self.goal_topic, self._on_goal, 10)
        self._global_path_sub = self.create_subscription(
            Path,
            self.global_path_topic,
            self._on_global_path,
            10,
        )
        self._timer = self.create_timer(1.0 / max(self.timer_rate, 0.1), self._on_timer)

        state = "enabled" if self.enabled else "disabled"
        self.get_logger().info(
            f"Robot target dataset logger {state}; robot={self.robot_topic}, humans={self.human_states_topic}, "
            f"goal={self.goal_topic}, require_goal={self.require_goal}, "
            f"global_path={self.global_path_topic}, "
            f"guidance_radius={self.guidance_point_radius:.2f}, "
            f"goal_reached_tolerance={self.goal_reached_tolerance:.2f}, output={self.output_path}"
        )

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _now_float(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_robot(self, msg: Agent) -> None:
        self._latest_robot = (float(msg.position.position.x), float(msg.position.position.y))
        self._last_robot_stamp = self._now_float()

    def _on_humans(self, msg: Agents) -> None:
        self._latest_humans = {
            int(agent.id): (float(agent.position.position.x), float(agent.position.position.y))
            for agent in msg.agents
        }
        self._last_humans_stamp = self._now_float()

    def _on_goal(self, msg: PoseStamped) -> None:
        # A new RViz goal starts a new collection episode. Keeping frames from
        # the previous goal would create 20-step windows whose past and future
        # belong to different navigation tasks.
        if self._goal_active:
            self._discard_pending_episode("replaced before reaching its goal")
        if self._has_received_goal:
            self._episode_id += 1
        self._frames.clear()
        self._last_record_stamp = None
        self._latest_goal = (float(msg.pose.position.x), float(msg.pose.position.y))
        self._latest_global_path = []
        self._last_global_path_stamp = None
        self._goal_active = True
        self._has_received_goal = True
        self._last_goal_stamp = self._now_float()
        self.get_logger().info(
            f"Started dataset episode {self._episode_id} from {self.goal_topic}: "
            f"({self._latest_goal[0]:.3f}, {self._latest_goal[1]:.3f})"
        )

    def _on_global_path(self, msg: Path) -> None:
        points = [
            (float(pose.pose.position.x), float(pose.pose.position.y))
            for pose in msg.poses
        ]
        if points:
            self._latest_global_path = points
            self._last_global_path_stamp = self._now_float()

    def _on_timer(self) -> None:
        if not self.enabled:
            return

        stamp = self._now_float()
        if stamp <= 0.0:
            return
        if (
            self._goal_active
            and self._last_goal_stamp is not None
            and self.episode_timeout > 0.0
            and stamp - self._last_goal_stamp >= self.episode_timeout
        ):
            self._discard_pending_episode(
                f"goal timeout after {self.episode_timeout:.1f} seconds"
            )
            self._goal_active = False
            self._flush(force=True)
            return
        record_dt = max(self.record_dt, 1e-3)
        if self._last_record_stamp is not None and stamp - self._last_record_stamp < record_dt:
            return
        if not self._has_fresh_state(stamp):
            return
        if self._goal_is_reached():
            pending_count = len(self._pending_samples)
            self._commit_pending_episode()
            self.get_logger().info(
                f"Closed dataset episode {self._episode_id}: robot reached goal within "
                f"{max(0.0, self.goal_reached_tolerance):.2f} m; "
                f"committed {pending_count} samples"
            )
            self._goal_active = False
            self._frames.clear()
            self._last_record_stamp = None
            self._flush(force=True)
            return
        if (
            self.max_samples > 0
            and len(self._samples) + len(self._pending_samples) >= self.max_samples
        ):
            return

        self._last_record_stamp = stamp
        self._frame_count += 1
        self._frames.append(
            {
                "stamp": stamp,
                "frame": self._frame_count,
                "robot": self._latest_robot,
                "humans": dict(self._latest_humans),
                "goal": self._latest_goal,
                "goal_stamp": self._last_goal_stamp,
                "global_path": list(self._latest_global_path),
                "global_path_stamp": self._last_global_path_stamp,
                "robot_state_age_s": max(
                    0.0,
                    stamp - float(self._last_robot_stamp),
                ),
                "human_state_age_s": max(
                    0.0,
                    stamp - float(self._last_humans_stamp),
                ),
            }
        )

        stride = max(self.sample_stride, 1)
        if len(self._frames) == self.seq_len and (self._frame_count - self.seq_len) % stride == 0:
            self._append_sample_from_window()

        self._flush(force=False)

    def _has_fresh_state(self, stamp: float) -> bool:
        if self._latest_robot is None or self._last_robot_stamp is None:
            return False
        if self._last_humans_stamp is None:
            return False
        if self.require_goal and self._latest_goal is None:
            return False
        if self.require_goal and not self._goal_active:
            return False
        if stamp - self._last_robot_stamp > self.stale_timeout:
            return False
        if stamp - self._last_humans_stamp > self.stale_timeout:
            return False
        return True

    def _goal_is_reached(self) -> bool:
        if not self._goal_active or self._latest_robot is None or self._latest_goal is None:
            return False
        tolerance = max(0.0, self.goal_reached_tolerance)
        if tolerance <= 0.0:
            return False
        return float(
            np.hypot(
                self._latest_goal[0] - self._latest_robot[0],
                self._latest_goal[1] - self._latest_robot[1],
            )
        ) <= tolerance

    def _append_sample_from_window(self) -> None:
        if (
            self.max_samples > 0
            and len(self._samples) + len(self._pending_samples) >= self.max_samples
        ):
            return

        frames = list(self._frames)
        human_ids = sorted(
            {
                human_id
                for frame in frames[: self.obs_len]
                for human_id in frame["humans"].keys()
                if all(human_id in obs_frame["humans"] for obs_frame in frames[: self.obs_len])
            }
        )
        if not human_ids:
            return

        trajs = np.full((1 + len(human_ids), self.seq_len, 2), np.nan, dtype=np.float32)

        for t_idx, frame in enumerate(frames):
            robot_pos = frame["robot"]
            if robot_pos is None:
                return
            trajs[0, t_idx, 0] = robot_pos[0]
            trajs[0, t_idx, 1] = robot_pos[1]

        if np.any(np.isnan(trajs[0, :, 0])):
            return

        for row, human_id in enumerate(human_ids, start=1):
            for t_idx, frame in enumerate(frames):
                pos = frame["humans"].get(human_id)
                if pos is None:
                    continue
                trajs[row, t_idx, 0] = pos[0]
                trajs[row, t_idx, 1] = pos[1]

        obs_end_frame = frames[self.obs_len - 1]
        frame_intervals = [
            float(current["stamp"] - previous["stamp"])
            for previous, current in zip(frames[:-1], frames[1:])
        ]
        final_goal = obs_end_frame.get("goal")
        goal_stamp = obs_end_frame.get("goal_stamp")
        final_goal_source = "rviz_goal_pose"
        if final_goal is None:
            if self.require_goal:
                return
            final_goal = tuple(trajs[0, -1].astype(np.float32))
            final_goal_source = "target_future_endpoint_fallback"
        current_xy = trajs[0, self.obs_len - 1]
        global_path = list(obs_end_frame.get("global_path") or [])
        global_path_stamp = obs_end_frame.get("global_path_stamp")
        guidance_point = None
        guidance_policy = "missing_global_path_recompute_in_postprocess"
        global_path_goal_distance = math.inf
        if global_path:
            global_path_goal_distance = float(
                np.hypot(
                    global_path[-1][0] - float(final_goal[0]),
                    global_path[-1][1] - float(final_goal[1]),
                )
            )
            if global_path_goal_distance <= max(
                0.0,
                self.global_path_goal_tolerance,
            ):
                try:
                    guidance_point = guidance_point_along_path(
                        global_path,
                        current=(float(current_xy[0]), float(current_xy[1])),
                        final_goal=(float(final_goal[0]), float(final_goal[1])),
                        radius=self.guidance_point_radius,
                    )
                    guidance_policy = "nav2_global_path_lookahead"
                except ValueError:
                    guidance_point = None

        sample_meta = {
            "episode_id": int(self._episode_id),
            "target": "robot",
            "target_row": 0,
            "neighbor_type": "humans",
            "human_ids": human_ids,
            "start_stamp": frames[0]["stamp"],
            "end_stamp": frames[-1]["stamp"],
            "start_frame": frames[0]["frame"],
            "end_frame": frames[-1]["frame"],
            "frame_intervals_s": frame_intervals,
            "mean_frame_interval_s": (
                float(np.mean(frame_intervals)) if frame_intervals else 0.0
            ),
            "max_frame_interval_error_s": (
                float(
                    np.max(
                        np.abs(
                            np.asarray(frame_intervals, dtype=np.float64)
                            - max(self.record_dt, 1e-3)
                        )
                    )
                )
                if frame_intervals
                else 0.0
            ),
            "max_robot_state_age_s": max(
                float(frame["robot_state_age_s"]) for frame in frames
            ),
            "max_human_state_age_s": max(
                float(frame["human_state_age_s"]) for frame in frames
            ),
            "scenario_name": self.scenario_name,
            "map_yaml_path": self.map_yaml_path,
            "robot_path_planner": self.robot_path_planner,
            "agent_motion_model": self.agent_motion_model,
            "pedestrians_avoid_robot": self.pedestrians_avoid_robot,
            "agents_wait_for_goal": self.agents_wait_for_goal,
            "collection_seed": self.collection_seed,
            "final_goal": [float(final_goal[0]), float(final_goal[1])],
            "final_goal_source": final_goal_source,
            "goal_topic": self.goal_topic,
            "goal_stamp": float(goal_stamp) if goal_stamp is not None else None,
            "global_path_topic": self.global_path_topic,
            "global_path_stamp": (
                float(global_path_stamp) if global_path_stamp is not None else None
            ),
            "global_path_pose_count": int(len(global_path)),
            "global_path_goal_distance_m": (
                float(global_path_goal_distance)
                if np.isfinite(global_path_goal_distance)
                else None
            ),
            "guidance_radius": float(max(0.0, self.guidance_point_radius)),
            "guidance_policy": guidance_policy,
        }
        if guidance_point is not None:
            sample_meta["guidance_point"] = [
                float(guidance_point[0]),
                float(guidance_point[1]),
            ]

        self._pending_samples.append(trajs)
        self._pending_meta.append(sample_meta)

    def _commit_pending_episode(self) -> None:
        if not self._pending_samples:
            return
        self._samples.extend(self._pending_samples)
        self._sample_meta.extend(self._pending_meta)
        self._pending_samples.clear()
        self._pending_meta.clear()

    def _discard_pending_episode(self, reason: str) -> None:
        count = len(self._pending_samples)
        self._pending_samples.clear()
        self._pending_meta.clear()
        self._frames.clear()
        self._last_record_stamp = None
        if count > 0:
            self.get_logger().warning(
                f"Discarded {count} samples from incomplete episode "
                f"{self._episode_id}: {reason}"
            )

    def _flush(self, *, force: bool) -> None:
        if not self.enabled:
            return
        if not force and self.flush_every <= 0:
            return
        if not force and len(self._samples) - self._last_flushed_sample_count < self.flush_every:
            return

        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        payload = {
            "all_trajs": self._samples,
            "all_scenes": [
                int(meta.get("episode_id", 0))
                for meta in self._sample_meta
            ],
            "scales": [1.0],
            "sample_meta": self._sample_meta,
            "metadata": {
                "format": "moai_robot_target_all_trajs_v2",
                "source": "hunavsim_robot_target_logger",
                "obs_len": self.obs_len,
                "pred_len": self.pred_len,
                "seq_len": self.seq_len,
                "dt": max(self.record_dt, 1e-3),
                "target": "robot",
                "target_row": 0,
                "neighbor_rows": "humans",
                "neighbor_future": "recorded_when_available",
                "robot_topic": self.robot_topic,
                "human_states_topic": self.human_states_topic,
                "goal_topic": self.goal_topic,
                "global_path_topic": self.global_path_topic,
                "scenario_name": self.scenario_name,
                "map_yaml_path": self.map_yaml_path,
                "robot_path_planner": self.robot_path_planner,
                "agent_motion_model": self.agent_motion_model,
                "pedestrians_avoid_robot": self.pedestrians_avoid_robot,
                "agents_wait_for_goal": self.agents_wait_for_goal,
                "collection_seed": self.collection_seed,
                "require_goal": self.require_goal,
                "goal_reached_tolerance": float(max(0.0, self.goal_reached_tolerance)),
                "episode_timeout": float(max(0.0, self.episode_timeout)),
                "final_goal_source": "rviz_goal_pose_when_available",
                "guidance_policy": "nav2_global_path_lookahead_when_available",
                "missing_guidance_policy": "recompute_from_inflated_map_in_postprocess",
                "guidance_radius": float(max(0.0, self.guidance_point_radius)),
                "sample_stride": max(self.sample_stride, 1),
                "stale_timeout": max(self.stale_timeout, 0.0),
                "description": "Each all_trajs item has the robot at row 0 and human neighbors in rows 1:.",
            },
        }
        tmp_path = f"{self.output_path}.tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, self.output_path)
        self._last_flushed_sample_count = len(self._samples)
        if force or self._samples:
            self.get_logger().info(f"Saved {len(self._samples)} robot-target samples to {self.output_path}")

    def close(self) -> None:
        if self._goal_active:
            self._discard_pending_episode("simulation stopped before goal completion")
            self._goal_active = False
        self._flush(force=True)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = JackalTeleopDatasetLoggerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
