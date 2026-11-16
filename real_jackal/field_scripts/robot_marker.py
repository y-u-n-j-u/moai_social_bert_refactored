#!/usr/bin/env python3
"""Display only: the robot's position RIGHT NOW in RViz (not the travelled path).

Publishes /robot_pose_marker (MarkerArray, frame ``map``) at 10 Hz from the TF map -> base_link: a disc with the footprint radius
(default now a small 0.30 m dot; 11-16 operator: the big body disc and the heading arrow were too much) and, only when show_arrow is true, an arrow for the heading. When the TF is older than ``max_age_s`` (localization lost, nothing running) the marker is
not refreshed and disappears (lifetime 0.5 s), so a stale position is never shown as if it were live. Publishes nothing the robot uses.
"""
import math

import tf2_ros

import rclpy
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


class RobotMarker(Node):
    def __init__(self):
        super().__init__("robot_marker")
        p = self.declare_parameter
        self.fixed_frame = p("fixed_frame", "map").value
        self.base_frame = p("base_frame", "base_link").value
        self.out_topic = p("out_topic", "/robot_pose_marker").value
        self.radius_m = float(p("radius_m", 0.15).value)
        self.show_arrow = bool(p("show_arrow", False).value)
        self.max_age_s = float(p("max_age_s", 1.0).value)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.pub = self.create_publisher(MarkerArray, self.out_topic, 1)
        self.create_timer(0.1, self.draw)

    def marker(self, mid, kind, color):
        m = Marker()
        m.header.frame_id = self.fixed_frame
        m.ns, m.id, m.type, m.action = "robot_now", mid, kind, Marker.ADD
        m.pose.orientation.w = 1.0
        m.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])
        m.lifetime.sec, m.lifetime.nanosec = 0, 500_000_000
        return m

    def draw(self):
        try:
            tf = self.tf_buffer.lookup_transform(self.fixed_frame, self.base_frame, rclpy.time.Time())
        except Exception:
            return
        age = (self.get_clock().now() - rclpy.time.Time.from_msg(tf.header.stamp)).nanoseconds * 1e-9
        if age > self.max_age_s:
            return
        t, q = tf.transform.translation, tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        body = self.marker(0, Marker.CYLINDER, (0.0, 0.85, 1.0, 0.55))
        body.pose.position.x, body.pose.position.y, body.pose.position.z = t.x, t.y, 0.03
        body.scale.x = body.scale.y = 2.0 * self.radius_m
        body.scale.z = 0.04
        markers = [body]
        if self.show_arrow:
            arrow = self.marker(1, Marker.ARROW, (1.0, 1.0, 1.0, 0.95))
            arrow.pose.position.x, arrow.pose.position.y, arrow.pose.position.z = t.x, t.y, 0.08
            arrow.pose.orientation.z, arrow.pose.orientation.w = math.sin(0.5 * yaw), math.cos(0.5 * yaw)
            arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.75, 0.12, 0.12
            markers.append(arrow)
        self.pub.publish(MarkerArray(markers=markers))


def main():
    rclpy.init()
    node = RobotMarker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
