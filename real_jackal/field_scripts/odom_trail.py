#!/usr/bin/env python3
"""Accumulate the robot's travelled odom positions into a Path for RViz (display only).

A drive is recorded from the moment the tracker leaves the stopped state until it is stopped again
(goal reached, e-stop, auto-stop). The finished line stays on screen; it is cleared when the next
drive starts, when a new goal is set, or when odom jumps (LIO restart).
"""
import math
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import String
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


IDLE_STATES = ("stopped", "goal_reached_stopped")


def is_idle(status: str) -> bool:
    return status.startswith(IDLE_STATES) or not status


class OdomTrail(Node):
    def __init__(self):
        super().__init__("odom_trail")
        self.path = Path()
        self.path.header.frame_id = "odom"
        self.last = None
        self.active = False
        self.clear_pending = True   # publish one empty Path so RViz drops any old line (also at start-up)
        status_topic = self.declare_parameter("status_topic", "/spu_bert/tracker_status").value
        trail_topic = self.declare_parameter("trail_topic", "/spu_bert/odom_trail").value
        self.pub = self.create_publisher(Path, trail_topic, 1)
        self.create_subscription(Odometry, "/aft_mapped_to_init", self.on_odom, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, "/goal_pose", self.on_goal, 1)  # a new goal starts a new trail
        self.create_subscription(String, status_topic, self.on_status, 10)
        self.create_timer(0.5, self.publish)

    def publish(self):
        if self.path.poses or self.clear_pending:
            self.pub.publish(self.path)
            self.clear_pending = False

    def clear(self):
        self.path.poses = []
        self.last = None
        self.clear_pending = True
        # an empty Path needs a valid frame, otherwise RViz drops it ("cannot transform") and keeps the old line
        self.path.header.frame_id = self.path.header.frame_id or "odom"

    def on_status(self, msg):
        running = not is_idle(msg.data)
        if running and not self.active:   # the drive starts: forget the previous one
            self.clear()
        self.active = running

    def on_goal(self, _msg):
        self.clear()

    def on_odom(self, msg):
        if not self.active:
            return
        p = msg.pose.pose.position
        if not all(math.isfinite(v) for v in (p.x, p.y)):
            return
        if self.last is not None and math.hypot(p.x - self.last[0], p.y - self.last[1]) < 0.05:
            return
        if self.last is not None and math.hypot(p.x - self.last[0], p.y - self.last[1]) > 5.0:
            self.clear()  # odom was reset (LIO restart): do not draw a line across the jump
        self.last = (p.x, p.y)
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose
        self.path.header = msg.header
        self.path.poses.append(pose)
        if len(self.path.poses) > 20000:
            del self.path.poses[:5000]


def main():
    rclpy.init()
    node = OdomTrail()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
