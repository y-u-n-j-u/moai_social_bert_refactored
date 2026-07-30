#!/usr/bin/env python3
from __future__ import annotations

import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


class CmdVelPassthroughNode(Node):
    """Forward Nav2 commands when the PMB2 velocity smoother drops output."""

    def __init__(self) -> None:
        super().__init__("cmd_vel_passthrough")
        input_topic = str(self.declare_parameter("input_topic", "/cmd_vel_nav").value)
        output_topic = str(self.declare_parameter("output_topic", "/cmd_vel").value)
        self._publisher = self.create_publisher(Twist, output_topic, 10)
        self._subscription = self.create_subscription(
            Twist, input_topic, self._publisher.publish, 10
        )
        self.get_logger().info(
            f"Forwarding PMB2 Nav2 velocity commands: {input_topic} -> {output_topic}"
        )


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
