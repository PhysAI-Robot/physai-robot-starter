"""Send one NavigateToPose goal through the ROS2 navigation graph."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import UInt32

from physai.robots.turtlebot.navigation import Nav2AcceptanceResult


class NavigateToPoseClient(Node):
    def __init__(self, robot: str, action_name: str, odom_topic: str,
                 collision_topic: str) -> None:
        super().__init__(f"{robot}_navigation_client")
        self.client = ActionClient(self, NavigateToPose, action_name)
        self.latest_odom: Odometry | None = None
        self.latest_collision_count: int | None = None
        self.report = Nav2AcceptanceResult(
            action_status=None,
            goal_accepted=False,
            timed_out=False,
            position_error=None,
            collision_count=None,
            failure_reason="not_started",
        )
        self.create_subscription(Odometry, odom_topic, self._receive_odom, 10)
        self.create_subscription(
            UInt32, collision_topic, self._receive_collision_count, 10
        )

    def _receive_odom(self, message: Odometry) -> None:
        self.latest_odom = message

    def _receive_collision_count(self, message: UInt32) -> None:
        self.latest_collision_count = int(message.data)

    def send_goal(self, x: float, y: float, yaw: float, max_position_error: float) -> int:
        if not self.client.wait_for_server(timeout_sec=10.0):
            self.report = Nav2AcceptanceResult(None, False, False, None, None, "action_server_unavailable")
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
            self.report = Nav2AcceptanceResult(None, False, False, None, None, "goal_rejected")
            self.get_logger().error("NavigateToPose goal was rejected")
            return 3

        self.get_logger().info("NavigateToPose goal accepted")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=120.0)
        if not result_future.done():
            self.report = Nav2AcceptanceResult(None, True, True, None, self.latest_collision_count, "action_timeout")
            self.get_logger().error("NavigateToPose timed out")
            return 6
        result = result_future.result()
        if result is None:
            self.report = Nav2AcceptanceResult(None, True, False, None, self.latest_collision_count, "missing_action_result")
            self.get_logger().error("NavigateToPose returned no result")
            return 4

        self.get_logger().info(f"NavigateToPose status: {result.status}")
        deadline = time.monotonic() + 2.0
        while self.latest_odom is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.latest_odom is None:
            self.report = Nav2AcceptanceResult(result.status, True, False, None, self.latest_collision_count, "missing_final_odometry")
            self.get_logger().error("No final odometry message received")
            return 7
        position = self.latest_odom.pose.pose.position
        error = math.hypot(position.x - x, position.y - y)
        self.get_logger().info(
            f"Final odom: x={position.x:.3f}, y={position.y:.3f}, "
            f"position_error={error:.3f}"
        )
        collision_count = self.latest_collision_count
        self.get_logger().info(f"MuJoCo non-ground collision count: {collision_count}")
        if collision_count is None:
            self.report = Nav2AcceptanceResult(result.status, True, False, error, None, "missing_collision_telemetry")
            self.get_logger().error("No MuJoCo collision count message received")
            return 9
        failure_reason = None
        if result.status != 4:
            failure_reason = "nav2_action_failed"
        elif collision_count != 0:
            failure_reason = "collision_detected"
        elif error > max_position_error:
            failure_reason = "position_tolerance_exceeded"
        self.report = Nav2AcceptanceResult(
            result.status, True, False, error, collision_count, failure_reason
        )
        if result.status != 4:
            return 5
        if collision_count != 0:
            return 10
        return 0 if error <= max_position_error else 8


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="turtlebot4")
    parser.add_argument("--action", default="navigate_to_pose")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--collision-topic", default="/simulation/collision_count")
    parser.add_argument("--x", type=float, default=1.0)
    parser.add_argument("--y", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--max-position-error", type=float, default=0.30)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    rclpy.init()
    node = NavigateToPoseClient(
        args.robot, args.action, args.odom_topic, args.collision_topic
    )
    try:
        code = node.send_goal(args.x, args.y, args.yaw, args.max_position_error)
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(
                json.dumps(node.report.as_dict(), indent=2), encoding="utf-8"
            )
            node.get_logger().info(f"navigation report -> {args.json_out}")
        return code
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())