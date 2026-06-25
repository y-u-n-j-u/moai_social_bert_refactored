#!/usr/bin/env python3
from __future__ import annotations

import os
import pickle
from collections import deque
from math import hypot
from typing import Any, Deque, Dict, List

import numpy as np
import rclpy
from hunav_msgs.msg import Agents
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


class PedestrianDatasetLoggerNode(Node):
    """Record HuNav pedestrian trajectories in MOAI/SPU-BERT all_trajs format.

    Each sliding window creates one sample per pedestrian target:
    row 0 is the target pedestrian with obs+future labels, and rows 1: are
    neighbor pedestrians with observed history only. Neighbor future is NaN,
    matching the dataset contract used by the SPU-BERT bridge.
    """

    def __init__(self) -> None:
        super().__init__("pedestrian_dataset_logger")

        self.enabled = self._as_bool(self.declare_parameter("enabled", False).value)
        self.human_states_topic = str(self.declare_parameter("human_states_topic", "/human_states").value)
        self.output_path = str(
            self.declare_parameter(
                "output_path",
                "/home/hunav_gz_classic_ws/moai_recordings/pedestrian_all_trajs.pkl",
            ).value
        )
        self.source_label = str(self.declare_parameter("source_label", "hunavsim_pedestrian_logger").value)
        self.obs_len = int(self.declare_parameter("obs_len", 8).value)
        self.pred_len = int(self.declare_parameter("pred_len", 12).value)
        self.record_dt = float(self.declare_parameter("record_dt", 0.4).value)
        self.sample_stride = int(self.declare_parameter("sample_stride", 1).value)
        self.flush_every = int(self.declare_parameter("flush_every", 10).value)
        self.max_samples = int(self.declare_parameter("max_samples", 0).value)
        self.guidance_point_radius = float(self.declare_parameter("guidance_point_radius", 8.0).value)

        self.seq_len = self.obs_len + self.pred_len
        self._last_record_stamp: float | None = None
        self._frame_count = 0
        self._last_flushed_sample_count = 0
        self._frames: Deque[Dict[str, Any]] = deque(maxlen=self.seq_len)
        self._samples: List[np.ndarray] = []
        self._sample_meta: List[Dict[str, Any]] = []

        self._sub = self.create_subscription(Agents, self.human_states_topic, self._on_humans, 10)
        state = "enabled" if self.enabled else "disabled"
        self.get_logger().info(
            f"Pedestrian dataset logger {state}; topic={self.human_states_topic}, output={self.output_path}"
        )

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @staticmethod
    def _stamp_to_float(msg: Agents) -> float:
        stamp = msg.header.stamp
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _on_humans(self, msg: Agents) -> None:
        if not self.enabled:
            return
        if self.max_samples > 0 and len(self._samples) >= self.max_samples:
            return

        stamp = self._stamp_to_float(msg)
        if stamp <= 0.0:
            stamp = self.get_clock().now().nanoseconds * 1e-9
        record_dt = max(self.record_dt, 1e-3)
        if self._last_record_stamp is not None and stamp - self._last_record_stamp < record_dt:
            return
        if self._last_record_stamp is not None and stamp < self._last_record_stamp:
            self._frames.clear()
            self._last_record_stamp = None

        agents = {
            int(agent.id): (
                float(agent.position.position.x),
                float(agent.position.position.y),
            )
            for agent in msg.agents
        }
        goals = {
            int(agent.id): (
                float(agent.goals[0].position.x),
                float(agent.goals[0].position.y),
            )
            for agent in msg.agents
            if agent.goals
        }
        if not agents:
            return

        self._last_record_stamp = stamp
        self._frame_count += 1
        self._frames.append({"stamp": stamp, "frame": self._frame_count, "agents": agents, "goals": goals})

        stride = max(self.sample_stride, 1)
        if len(self._frames) == self.seq_len and (self._frame_count - self.seq_len) % stride == 0:
            self._append_samples_from_window()

        self._flush(force=False)

    def _append_samples_from_window(self) -> None:
        if self.max_samples > 0 and len(self._samples) >= self.max_samples:
            return

        frames = list(self._frames)
        target_ids = sorted({agent_id for frame in frames for agent_id in frame["agents"].keys()})

        for target_id in target_ids:
            if self.max_samples > 0 and len(self._samples) >= self.max_samples:
                break
            if not all(target_id in frame["agents"] for frame in frames):
                continue

            neighbor_ids = [
                agent_id
                for agent_id in target_ids
                if agent_id != target_id and all(agent_id in frame["agents"] for frame in frames[: self.obs_len])
            ]
            agent_ids = [target_id] + neighbor_ids
            trajs = np.full((len(agent_ids), self.seq_len, 2), np.nan, dtype=np.float32)

            for row, agent_id in enumerate(agent_ids):
                limit = self.seq_len if row == 0 else self.obs_len
                for t_idx, frame in enumerate(frames[:limit]):
                    pos = frame["agents"].get(agent_id)
                    if pos is None:
                        continue
                    trajs[row, t_idx, 0] = pos[0]
                    trajs[row, t_idx, 1] = pos[1]

            if np.any(np.isnan(trajs[0, :, 0])):
                continue

            obs_end = frames[self.obs_len - 1]
            current_xy = trajs[0, self.obs_len - 1].astype(np.float32)
            final_goal_xy = obs_end.get("goals", {}).get(target_id)
            if final_goal_xy is None:
                final_goal_xy = tuple(trajs[0, -1].astype(np.float32))
            guidance_xy = self._guidance_point_from_goal(current_xy, final_goal_xy)

            self._samples.append(trajs)
            self._sample_meta.append(
                {
                    "target_id": target_id,
                    "agent_ids": agent_ids,
                    "start_stamp": frames[0]["stamp"],
                    "end_stamp": frames[-1]["stamp"],
                    "start_frame": frames[0]["frame"],
                    "end_frame": frames[-1]["frame"],
                    "final_goal": [float(final_goal_xy[0]), float(final_goal_xy[1])],
                    "guidance_point": [float(guidance_xy[0]), float(guidance_xy[1])],
                    "guidance_radius": float(max(0.0, self.guidance_point_radius)),
                    "guidance_policy": "circle_line_intersection_to_final_goal",
                }
            )

    def _guidance_point_from_goal(
        self,
        current_xy: np.ndarray,
        final_goal_xy: tuple[float, float],
    ) -> tuple[float, float]:
        cx = float(current_xy[0])
        cy = float(current_xy[1])
        gx = float(final_goal_xy[0])
        gy = float(final_goal_xy[1])
        dx = gx - cx
        dy = gy - cy
        dist = hypot(dx, dy)
        if dist <= 1e-6:
            return gx, gy
        step = min(max(0.0, self.guidance_point_radius), dist)
        return cx + dx / dist * step, cy + dy / dist * step

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
            "all_scenes": [0] * len(self._samples),
            "scales": [1.0],
            "sample_meta": self._sample_meta,
            "metadata": {
                "format": "moai_all_trajs_v1",
                "source": self.source_label,
                "obs_len": self.obs_len,
                "pred_len": self.pred_len,
                "seq_len": self.seq_len,
                "dt": max(self.record_dt, 1e-3),
                "neighbor_future": "nan",
                "guidance_policy": "circle_line_intersection_to_final_goal",
                "guidance_radius": float(max(0.0, self.guidance_point_radius)),
                "human_states_topic": self.human_states_topic,
                "description": "Each all_trajs item has target at row 0 and neighbors in rows 1:.",
            },
        }
        tmp_path = f"{self.output_path}.tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, self.output_path)
        self._last_flushed_sample_count = len(self._samples)
        if force or self._samples:
            self.get_logger().info(f"Saved {len(self._samples)} pedestrian samples to {self.output_path}")

    def close(self) -> None:
        self._flush(force=True)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PedestrianDatasetLoggerNode()
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
