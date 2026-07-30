#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


Pose2D = Tuple[float, float, float]


def _yaw_from_odometry(msg: Odometry) -> float:
    q = msg.pose.pose.orientation
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


class GroundTruthLocalizationNode(Node):
    """Publish drift-free map->odom using Gazebo truth and wheel odometry.

    PMB2 still keeps its normal odom->base_footprint transform. This node
    computes the missing transform as:

        T_map_odom = T_map_base(truth) * inverse(T_odom_base(wheel))

    The result lets Nav2 use the exact simulated pose without changing the
    robot controller or the dataset's map coordinate system.
    """

    def __init__(self) -> None:
        super().__init__("ground_truth_localization")
        self.truth_topic = str(
            self.declare_parameter("truth_topic", "/ground_truth_odom").value
        )
        self.wheel_odom_topic = str(
            self.declare_parameter(
                "wheel_odom_topic", "/mobile_base_controller/odom"
            ).value
        )
        self.map_frame = str(self.declare_parameter("map_frame", "map").value)
        self.odom_frame = str(self.declare_parameter("odom_frame", "odom").value)

        self._wheel_pose: Optional[Pose2D] = None
        self._broadcaster = TransformBroadcaster(self)
        self._wheel_sub = self.create_subscription(
            Odometry, self.wheel_odom_topic, self._on_wheel_odom, 20
        )
        self._truth_sub = self.create_subscription(
            Odometry, self.truth_topic, self._on_truth, 20
        )
        self.get_logger().info(
            "Ground-truth localization enabled: "
            f"{self.truth_topic} + {self.wheel_odom_topic} -> "
            f"{self.map_frame}->{self.odom_frame}"
        )

    def _on_wheel_odom(self, msg: Odometry) -> None:
        self._wheel_pose = (
            float(msg.pose.pose.position.x),
            float(msg.pose.pose.position.y),
            _yaw_from_odometry(msg),
        )

    def _on_truth(self, msg: Odometry) -> None:
        wheel = self._wheel_pose
        if wheel is None:
            return

        truth_x = float(msg.pose.pose.position.x)
        truth_y = float(msg.pose.pose.position.y)
        truth_yaw = _yaw_from_odometry(msg)
        odom_x, odom_y, odom_yaw = wheel

        map_to_odom_yaw = math.atan2(
            math.sin(truth_yaw - odom_yaw),
            math.cos(truth_yaw - odom_yaw),
        )
        cosine = math.cos(map_to_odom_yaw)
        sine = math.sin(map_to_odom_yaw)
        map_to_odom_x = truth_x - (cosine * odom_x - sine * odom_y)
        map_to_odom_y = truth_y - (sine * odom_x + cosine * odom_y)

        transform = TransformStamped()
        transform.header.stamp = msg.header.stamp
        transform.header.frame_id = self.map_frame
        transform.child_frame_id = self.odom_frame
        transform.transform.translation.x = map_to_odom_x
        transform.transform.translation.y = map_to_odom_y
        transform.transform.rotation.z = math.sin(0.5 * map_to_odom_yaw)
        transform.transform.rotation.w = math.cos(0.5 * map_to_odom_yaw)
        self._broadcaster.sendTransform(transform)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GroundTruthLocalizationNode()
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
