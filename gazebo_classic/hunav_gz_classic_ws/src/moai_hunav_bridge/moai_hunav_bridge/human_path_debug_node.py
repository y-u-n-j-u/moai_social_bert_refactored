#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from math import sqrt
from typing import Deque, Dict, Iterable, List, Tuple

import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from hunav_msgs.msg import Agents
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


@dataclass(frozen=True)
class TrackPoint:
    stamp: float
    x: float
    y: float
    z: float
    vx: float
    vy: float


class HumanPathDebugNode(Node):
    def __init__(self) -> None:
        super().__init__("human_path_debug_node")

        self.human_states_topic = self.declare_parameter("human_states_topic", "/human_states").value
        self.marker_topic = self.declare_parameter("marker_topic", "/moai/debug_human_paths").value
        self.obs_len = int(self.declare_parameter("obs_len", 8).value)
        self.pred_len = int(self.declare_parameter("pred_len", 12).value)
        self.prediction_dt = float(self.declare_parameter("prediction_dt", 0.4).value)
        self.history_timeout = float(self.declare_parameter("history_timeout", 3.0).value)
        self.publish_rate = float(self.declare_parameter("publish_rate", 5.0).value)

        self._tracks: Dict[int, Deque[TrackPoint]] = defaultdict(lambda: deque(maxlen=self.obs_len))
        self._frame_id = "map"
        self._last_stamp = self.get_clock().now()
        self._last_msg_stamp = 0.0

        self._sub = self.create_subscription(Agents, self.human_states_topic, self._on_human_states, 10)
        self._pub = self.create_publisher(MarkerArray, self.marker_topic, 10)
        self._timer = self.create_timer(1.0 / max(self.publish_rate, 0.1), self._publish_markers)

        self.get_logger().info(
            f"Listening to {self.human_states_topic}; publishing trajectory markers on {self.marker_topic}"
        )

    def _on_human_states(self, msg: Agents) -> None:
        self._frame_id = msg.header.frame_id or self._frame_id
        self._last_stamp = self.get_clock().now()
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        if stamp <= 0.0:
            stamp = self._last_stamp.nanoseconds * 1e-9
        self._last_msg_stamp = stamp

        seen_ids = set()
        for agent in msg.agents:
            seen_ids.add(agent.id)
            position = agent.position.position
            velocity = agent.velocity.linear
            self._tracks[agent.id].append(
                TrackPoint(
                    stamp=stamp,
                    x=float(position.x),
                    y=float(position.y),
                    z=float(position.z),
                    vx=float(velocity.x),
                    vy=float(velocity.y),
                )
            )

        stale_ids = [
            agent_id
            for agent_id, track in self._tracks.items()
            if not track or self._last_msg_stamp - track[-1].stamp > self.history_timeout
        ]
        for agent_id in stale_ids:
            del self._tracks[agent_id]

    def _publish_markers(self) -> None:
        markers = MarkerArray()
        markers.markers.append(self._delete_all_marker())

        marker_id = 1
        for agent_id in sorted(self._tracks):
            track = list(self._tracks[agent_id])
            if not track:
                continue

            observed = self._points_from_track(track)
            predicted = self._predict_points(track)

            markers.markers.append(
                self._line_marker(
                    marker_id,
                    f"agent_{agent_id}_observed",
                    observed,
                    ColorRGBA(r=0.1, g=0.7, b=1.0, a=0.95),
                    scale=0.09,
                )
            )
            marker_id += 1
            markers.markers.append(
                self._points_marker(
                    marker_id,
                    f"agent_{agent_id}_observed_points",
                    observed,
                    ColorRGBA(r=0.1, g=0.7, b=1.0, a=0.95),
                    scale=0.18,
                )
            )
            marker_id += 1
            markers.markers.append(
                self._line_marker(
                    marker_id,
                    f"agent_{agent_id}_predicted",
                    predicted,
                    ColorRGBA(r=1.0, g=0.35, b=0.05, a=0.9),
                    scale=0.055,
                )
            )
            marker_id += 1
            markers.markers.append(
                self._label_marker(
                    marker_id,
                    f"agent_{agent_id}_label",
                    track[-1],
                    f"agent {agent_id}",
                )
            )
            marker_id += 1

        self._pub.publish(markers)

    def _delete_all_marker(self) -> Marker:
        marker = Marker()
        marker.header.frame_id = self._frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.action = Marker.DELETEALL
        return marker

    def _line_marker(
        self,
        marker_id: int,
        namespace: str,
        points: Iterable[Point],
        color: ColorRGBA,
        scale: float,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = self._frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = scale
        marker.color = color
        marker.lifetime = Duration(sec=1)
        marker.points = list(points)
        return marker

    def _points_marker(
        self,
        marker_id: int,
        namespace: str,
        points: Iterable[Point],
        color: ColorRGBA,
        scale: float,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = self._frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        marker.color = color
        marker.lifetime = Duration(sec=1)
        marker.points = list(points)
        return marker

    def _label_marker(self, marker_id: int, namespace: str, point: TrackPoint, text: str) -> Marker:
        marker = Marker()
        marker.header.frame_id = self._frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = point.x
        marker.pose.position.y = point.y
        marker.pose.position.z = max(point.z, 0.1) + 0.35
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.32
        marker.color = ColorRGBA(r=0.0, g=0.0, b=0.0, a=0.95)
        marker.lifetime = Duration(sec=1)
        marker.text = text
        return marker

    def _points_from_track(self, track: List[TrackPoint]) -> List[Point]:
        return [Point(x=p.x, y=p.y, z=max(p.z, 0.05) + 0.08) for p in track]

    def _predict_points(self, track: List[TrackPoint]) -> List[Point]:
        last = track[-1]
        vx, vy = self._estimate_velocity(track)
        z = max(last.z, 0.05) + 0.18
        points = [Point(x=last.x, y=last.y, z=z)]
        for step in range(1, self.pred_len + 1):
            dt = self.prediction_dt * step
            points.append(Point(x=last.x + vx * dt, y=last.y + vy * dt, z=z))
        return points

    def _estimate_velocity(self, track: List[TrackPoint]) -> Tuple[float, float]:
        last = track[-1]
        speed = sqrt(last.vx * last.vx + last.vy * last.vy)
        if speed > 0.01:
            return last.vx, last.vy

        if len(track) < 2:
            return 0.0, 0.0

        prev = track[-2]
        dt = max(last.stamp - prev.stamp, 1e-3)
        return (last.x - prev.x) / dt, (last.y - prev.y) / dt


def main(args: List[str] | None = None) -> None:
    rclpy.init(args=args)
    node = HumanPathDebugNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
