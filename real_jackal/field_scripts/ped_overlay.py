#!/usr/bin/env python3
"""Display only: show what the planner sees of each tracked pedestrian (never publishes anything the robot uses).

For every track on /ped_tracking it draws
  * a red ring with the radius at which a candidate path is REJECTED (minimum_human_center_distance, 1.20 m), in the frame of the Tracks message
  * an orange ring with the radius of the tracker's hard STOP (human_stop_distance, 0.90 m), same frame
  * a label "id  v" with the tracker's speed
  * the TRAIL the person has walked (magenta line in the map frame, like the robot's odom trail), kept trail_seconds
    after the last sample of that id
No prediction is drawn: SPU-BERT only takes the pedestrian histories as input and outputs the robot path.
"""
import math
from collections import deque

import tf2_ros

import rclpy
from geometry_msgs.msg import Point
from moai_nav_msgs.msg import Tracks
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


def estimate_velocity(history, sample_dt, maximum_speed=2.5):
    """Same rule as navigation_core.estimate_velocity (kept as a copy so this file has no dependency)."""
    if len(history) < 2 or sample_dt <= 0.0:
        return 0.0, 0.0
    latest = history[-1]
    first = max(0, len(history) - 4)
    elapsed = max((len(history) - 1 - first) * sample_dt, sample_dt)
    vx = (latest[0] - history[first][0]) / elapsed
    vy = (latest[1] - history[first][1]) / elapsed
    speed = math.hypot(vx, vy)
    if speed > maximum_speed > 0.0:
        vx, vy = vx * maximum_speed / speed, vy * maximum_speed / speed
    return vx, vy


class PedOverlay(Node):
    def __init__(self):
        super().__init__("ped_overlay")
        p = self.declare_parameter
        self.tracks_topic = p("tracks_topic", "/ped_tracking").value
        self.out_topic = p("out_topic", "/ped_overlay").value
        self.reject_m = float(p("reject_radius_m", 1.20).value)
        self.stop_m = float(p("stop_radius_m", 0.90).value)
        self.dt = float(p("sample_dt", 0.4).value)
        self.steps = int(p("pred_steps", 12).value)
        self.obs_len = int(p("obs_len", 8).value)
        self.trail_frame = p("trail_frame", "map").value
        self.trail_seconds = float(p("trail_seconds", 60.0).value)
        self.trail_dt = float(p("trail_dt", 0.2).value)
        self.trails = {}
        self.trail_seen = {}
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.histories = {}
        self.latest = {}
        self.frame = "base_link"
        self.stamp = None
        self.pub = self.create_publisher(MarkerArray, self.out_topic, 1)
        self.create_subscription(Tracks, self.tracks_topic, self.on_tracks, qos_profile_sensor_data)
        self.create_timer(self.dt, self.sample)        # the bridge samples its histories the same way
        self.create_timer(0.2, self.draw)
        self.create_timer(self.trail_dt, self.sample_trail)

    def on_tracks(self, msg):
        self.frame = msg.header.frame_id or self.frame
        self.latest = {
            int(t.id): (float(t.pose.position.x), float(t.pose.position.y),
                        math.hypot(t.twist.linear.x, t.twist.linear.y))
            for t in msg.tracks
        }

    def sample(self):
        for tid, (x, y, _) in self.latest.items():
            self.histories.setdefault(tid, deque(maxlen=self.obs_len)).append((x, y))
        for tid in set(self.histories) - set(self.latest):
            self.histories.pop(tid, None)

    def sample_trail(self):
        """Append every tracked position, transformed to the trail frame, to that id's trail."""
        try:
            tf = self.tf_buffer.lookup_transform(self.trail_frame, self.frame, rclpy.time.Time())
        except Exception:
            return
        t, q = tf.transform.translation, tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        c, sn = math.cos(yaw), math.sin(yaw)
        now = self.get_clock().now().nanoseconds * 1e-9
        for tid, (x, y, _) in self.latest.items():
            trail = self.trails.setdefault(tid, deque(maxlen=int(self.trail_seconds / self.trail_dt)))
            trail.append((t.x + c * x - sn * y, t.y + sn * x + c * y))
            self.trail_seen[tid] = now
        for tid in [i for i, seen in self.trail_seen.items() if now - seen > 5.0]:
            self.trails.pop(tid, None)
            self.trail_seen.pop(tid, None)

    @staticmethod
    def marker(ns, mid, kind, color, scale, frame):
        m = Marker()
        m.header.frame_id = frame
        m.ns, m.id, m.type, m.action = ns, mid, kind, Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = scale
        m.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])
        m.lifetime.sec, m.lifetime.nanosec = 0, 600_000_000
        return m

    def ring(self, ns, mid, cx, cy, radius, color, frame):
        m = self.marker(ns, mid, Marker.LINE_STRIP, color, 0.04, frame)
        m.scale.y = m.scale.z = 0.0
        for k in range(41):
            a = 2.0 * math.pi * k / 40.0
            m.points.append(Point(x=cx + radius * math.cos(a), y=cy + radius * math.sin(a), z=0.05))
        return m

    def draw(self):
        array = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        array.markers.append(clear)
        for tid, (x, y, v_tracker) in self.latest.items():
            array.markers.append(self.ring("reject_1.2m", tid, x, y, self.reject_m, (1.0, 0.1, 0.1, 0.9), self.frame))
            array.markers.append(self.ring("stop_0.9m", tid, x, y, self.stop_m, (1.0, 0.55, 0.0, 0.9), self.frame))
            label = self.marker("label", tid, Marker.TEXT_VIEW_FACING, (1.0, 1.0, 1.0, 1.0), 0.16, self.frame)
            label.scale.x = label.scale.y = 0.0
            label.pose.position.x, label.pose.position.y, label.pose.position.z = x, y, 1.1
            label.text = "id %d  v %.2f m/s" % (tid, v_tracker)
            array.markers.append(label)
        for tid, trail in self.trails.items():
            if len(trail) < 2:
                continue
            line = self.marker("trail", tid, Marker.LINE_STRIP, (0.9, 0.2, 0.9, 0.95), 0.06, self.trail_frame)
            line.scale.y = line.scale.z = 0.0
            line.lifetime.sec, line.lifetime.nanosec = 1, 0
            line.points = [Point(x=px, y=py, z=0.08) for px, py in trail]
            array.markers.append(line)
        self.pub.publish(array)


def main():
    rclpy.init()
    node = PedOverlay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
