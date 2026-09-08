"""Send one NavigateToPose goal to the TurtleBot4 Nav2 stack."""

from __future__ import annotations

import argparse
import math
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node


class NavigateToPoseClient(Node):
    def __init__(self) -> None:
        super().__init__("turtlebot4_nav_goal_client")
        self.client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.latest_odom: Odometry | None = None
        self.create_subscription(Odometry, "/odom", self._receive_odom, 10)

    def _receive_odom(self, message: Odometry) -> None:
        self.latest_odom = message

    def send_goal(self, x: float, y: float, yaw: float, max_position_error: float) -> int:
        if not self.client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("NavigateToPose action server is unavailable")
            return 2

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)

        send_future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("NavigateToPose goal was rejected")
            return 3

        self.get_logger().info("NavigateToPose goal accepted")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=120.0)
        if not result_future.done():
            self.get_logger().error("NavigateToPose timed out")
            return 6
        result = result_future.result()
        if result is None:
            self.get_logger().error("NavigateToPose returned no result")
            return 4

        self.get_logger().info(f"NavigateToPose status: {result.status}")
        deadline = time.monotonic() + 2.0
        while self.latest_odom is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.latest_odom is None:
            self.get_logger().error("No final odometry message received")
            return 7
        position = self.latest_odom.pose.pose.position
        error = math.hypot(position.x - x, position.y - y)
        self.get_logger().info(
            f"Final odom: x={position.x:.3f}, y={position.y:.3f}, "
            f"position_error={error:.3f}"
        )
        if result.status != 4:
            return 5
        return 0 if error <= max_position_error else 8


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=float, default=1.0)
    parser.add_argument("--y", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--max-position-error", type=float, default=0.30)
    args = parser.parse_args()

    rclpy.init()
    node = NavigateToPoseClient()
    try:
        return node.send_goal(args.x, args.y, args.yaw, args.max_position_error)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())