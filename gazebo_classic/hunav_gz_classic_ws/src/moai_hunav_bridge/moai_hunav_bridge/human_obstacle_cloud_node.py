#!/usr/bin/env python3
from __future__ import annotations

from math import cos, pi, sin
from typing import Dict, List, Tuple

import rclpy
from hunav_msgs.msg import Agents
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray


class HumanObstacleCloudNode(Node):
    """Publish HuNav humans and predicted paths as a PointCloud2 obstacle source."""

    def __init__(self) -> None:
        super().__init__("human_obstacle_cloud")

        self.human_states_topic = str(self.declare_parameter("human_states_topic", "/human_states").value)
        self.predicted_paths_topic = str(
            self.declare_parameter("predicted_paths_topic", "/moai/social_bert_predicted_paths").value
        )
        self.cloud_topic = str(self.declare_parameter("cloud_topic", "/moai/human_obstacle_cloud").value)
        self.publish_rate = float(self.declare_parameter("publish_rate", 10.0).value)
        self.human_z = float(self.declare_parameter("human_z", 0.35).value)
        self.predicted_z = float(self.declare_parameter("predicted_z", 0.25).value)
        self.current_ring_points = int(self.declare_parameter("current_ring_points", 8).value)
        self.predicted_stride = int(self.declare_parameter("predicted_stride", 2).value)
        self.predicted_horizon_points = int(self.declare_parameter("predicted_horizon_points", 8).value)

        self._last_humans: Agents | None = None
        self._predicted_paths: Dict[int, List[Tuple[float, float]]] = {}

        self._human_sub = self.create_subscription(Agents, self.human_states_topic, self._on_humans, 10)
        self._pred_sub = self.create_subscription(MarkerArray, self.predicted_paths_topic, self._on_markers, 10)
        self._pub = self.create_publisher(PointCloud2, self.cloud_topic, 10)
        self._timer = self.create_timer(1.0 / max(self.publish_rate, 0.1), self._publish_cloud)

        self.get_logger().info(
            f"Publishing human obstacle cloud on {self.cloud_topic} from {self.human_states_topic}"
        )

    def _on_humans(self, msg: Agents) -> None:
        self._last_humans = msg

    def _on_markers(self, msg: MarkerArray) -> None:
        predicted: Dict[int, List[Tuple[float, float]]] = {}
        for marker in msg.markers:
            if marker.action == Marker.DELETEALL or "predicted_path" not in marker.ns:
                continue
            agent_id = self._agent_id_from_ns(marker.ns)
            if agent_id is None:
                continue
            predicted[agent_id] = [(float(point.x), float(point.y)) for point in marker.points[1:]]
        if predicted:
            self._predicted_paths = predicted

    def _publish_cloud(self) -> None:
        if self._last_humans is None:
            return

        frame_id = self._last_humans.header.frame_id or "map"
        stamp = self.get_clock().now().to_msg()
        points: List[Tuple[float, float, float]] = []

        for agent in self._last_humans.agents:
            ax = float(agent.position.position.x)
            ay = float(agent.position.position.y)
            radius = max(float(agent.radius), 0.35)
            points.append((ax, ay, self.human_z))
            ring_count = max(self.current_ring_points, 0)
            for idx in range(ring_count):
                angle = 2.0 * pi * idx / max(ring_count, 1)
                points.append((ax + radius * cos(angle), ay + radius * sin(angle), self.human_z))

            path = self._predicted_paths.get(int(agent.id), [])
            stride = max(self.predicted_stride, 1)
            horizon = max(self.predicted_horizon_points, 0)
            for px, py in path[::stride][:horizon]:
                points.append((px, py, self.predicted_z))

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        header = self._last_humans.header
        header.stamp = stamp
        header.frame_id = frame_id
        self._pub.publish(point_cloud2.create_cloud(header, fields, points))

    @staticmethod
    def _agent_id_from_ns(ns: str) -> int | None:
        try:
            # Expected form: agent_3_predicted_path
            return int(ns.split("_")[1])
        except (IndexError, ValueError):
            return None


def main(args: List[str] | None = None) -> None:
    rclpy.init(args=args)
    node = HumanObstacleCloudNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
