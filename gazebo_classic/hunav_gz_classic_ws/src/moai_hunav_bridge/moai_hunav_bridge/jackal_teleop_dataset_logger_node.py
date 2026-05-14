#!/usr/bin/env python3
from __future__ import annotations

import os
import pickle
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import rclpy
from hunav_msgs.msg import Agent, Agents
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


class JackalTeleopDatasetLoggerNode(Node):
    """Record teleoperated Jackal trajectories in MOAI all_trajs format.

    Row 0 is always the Jackal target. Rows 1: are humans used as neighbor
    context. The Jackal has obs+future positions, while human futures are NaN
    to match the SPU-BERT target/neighbors contract.
    """

    def __init__(self) -> None:
        super().__init__("jackal_teleop_dataset_logger")

        self.enabled = self._as_bool(self.declare_parameter("enabled", False).value)
        self.robot_topic = str(self.declare_parameter("robot_topic", "/robot_states").value)
        self.human_states_topic = str(self.declare_parameter("human_states_topic", "/human_states").value)
        self.output_path = str(
            self.declare_parameter(
                "output_path",
                "/home/hunav_gz_classic_ws/moai_recordings/jackal_teleop_all_trajs.pkl",
            ).value
        )
        self.obs_len = int(self.declare_parameter("obs_len", 8).value)
        self.pred_len = int(self.declare_parameter("pred_len", 12).value)
        self.record_dt = float(self.declare_parameter("record_dt", 0.4).value)
        self.sample_stride = int(self.declare_parameter("sample_stride", 1).value)
        self.flush_every = int(self.declare_parameter("flush_every", 10).value)
        self.max_samples = int(self.declare_parameter("max_samples", 0).value)
        self.stale_timeout = float(self.declare_parameter("stale_timeout", 2.0).value)
        self.timer_rate = float(self.declare_parameter("timer_rate", 20.0).value)

        self.seq_len = self.obs_len + self.pred_len
        self._latest_robot: Optional[Tuple[float, float]] = None
        self._latest_humans: Dict[int, Tuple[float, float]] = {}
        self._last_robot_stamp: Optional[float] = None
        self._last_humans_stamp: Optional[float] = None
        self._last_record_stamp: Optional[float] = None
        self._frame_count = 0
        self._last_flushed_sample_count = 0
        self._frames: Deque[Dict[str, Any]] = deque(maxlen=self.seq_len)
        self._samples: List[np.ndarray] = []
        self._sample_meta: List[Dict[str, Any]] = []

        self._robot_sub = self.create_subscription(Agent, self.robot_topic, self._on_robot, 10)
        self._humans_sub = self.create_subscription(Agents, self.human_states_topic, self._on_humans, 10)
        self._timer = self.create_timer(1.0 / max(self.timer_rate, 0.1), self._on_timer)

        state = "enabled" if self.enabled else "disabled"
        self.get_logger().info(
            f"Jackal teleop dataset logger {state}; robot={self.robot_topic}, humans={self.human_states_topic}, "
            f"output={self.output_path}"
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

    def _on_timer(self) -> None:
        if not self.enabled:
            return
        if self.max_samples > 0 and len(self._samples) >= self.max_samples:
            return

        stamp = self._now_float()
        if stamp <= 0.0:
            return
        record_dt = max(self.record_dt, 1e-3)
        if self._last_record_stamp is not None and stamp - self._last_record_stamp < record_dt:
            return
        if not self._has_fresh_state(stamp):
            return

        self._last_record_stamp = stamp
        self._frame_count += 1
        self._frames.append(
            {
                "stamp": stamp,
                "frame": self._frame_count,
                "robot": self._latest_robot,
                "humans": dict(self._latest_humans),
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
        if stamp - self._last_robot_stamp > self.stale_timeout:
            return False
        if stamp - self._last_humans_stamp > self.stale_timeout:
            return False
        return True

    def _append_sample_from_window(self) -> None:
        if self.max_samples > 0 and len(self._samples) >= self.max_samples:
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
            for t_idx, frame in enumerate(frames[: self.obs_len]):
                pos = frame["humans"].get(human_id)
                if pos is None:
                    continue
                trajs[row, t_idx, 0] = pos[0]
                trajs[row, t_idx, 1] = pos[1]

        self._samples.append(trajs)
        self._sample_meta.append(
            {
                "target": "jackal",
                "target_row": 0,
                "neighbor_type": "humans",
                "human_ids": human_ids,
                "start_stamp": frames[0]["stamp"],
                "end_stamp": frames[-1]["stamp"],
                "start_frame": frames[0]["frame"],
                "end_frame": frames[-1]["frame"],
            }
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
            "all_scenes": [0] * len(self._samples),
            "scales": [1.0],
            "sample_meta": self._sample_meta,
            "metadata": {
                "format": "moai_jackal_target_all_trajs_v1",
                "source": "hunavsim_jackal_teleop",
                "obs_len": self.obs_len,
                "pred_len": self.pred_len,
                "seq_len": self.seq_len,
                "dt": max(self.record_dt, 1e-3),
                "target": "jackal",
                "target_row": 0,
                "neighbor_rows": "humans",
                "neighbor_future": "nan",
                "robot_topic": self.robot_topic,
                "human_states_topic": self.human_states_topic,
                "description": "Each all_trajs item has Jackal at row 0 and human neighbors in rows 1:.",
            },
        }
        tmp_path = f"{self.output_path}.tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, self.output_path)
        self._last_flushed_sample_count = len(self._samples)
        if force or self._samples:
            self.get_logger().info(f"Saved {len(self._samples)} Jackal-target samples to {self.output_path}")

    def close(self) -> None:
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
