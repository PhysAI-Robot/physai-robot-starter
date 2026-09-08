"""ROS2 node adapter for the TurtleBot4 MuJoCo backend."""

from __future__ import annotations

import time
from typing import Any

from ...contracts import Action, Twist, Vector3
from .env import TurtleBot4Config, TurtleBot4Env


class TurtleBot4ROS2Node:
    """Run TurtleBot4 through real ROS2 command and state topics."""

    def __init__(self, node: Any, config: TurtleBot4Config | None = None) -> None:
        from geometry_msgs.msg import TransformStamped
        from geometry_msgs.msg import Twist as ROSTwist
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import JointState
        from tf2_msgs.msg import TFMessage

        self.node = node
        self.simulation = TurtleBot4Env(config)
        self._odom_type = Odometry
        self._joint_state_type = JointState
        self._tf_message_type = TFMessage
        self._transform_type = TransformStamped
        self._command = Twist()
        self._cmd_subscription = node.create_subscription(
            ROSTwist, "/cmd_vel", self._receive_twist, 10
        )
        self._joint_state_publisher = node.create_publisher(
            JointState, "/joint_states", 10
        )
        self._odom_publisher = node.create_publisher(Odometry, "/odom", 10)
        self._tf_publisher = node.create_publisher(TFMessage, "/tf", 10)

    def _receive_twist(self, message: Any) -> None:
        self._command = Twist(
            linear=Vector3(
                x=float(message.linear.x),
                y=float(message.linear.y),
                z=float(message.linear.z),
            ),
            angular=Vector3(
                x=float(message.angular.x),
                y=float(message.angular.y),
                z=float(message.angular.z),
            ),
            frame_id="base",
        )

    def _stamp(self, header: Any, value: float) -> None:
        seconds = max(0.0, float(value))
        header.stamp.sec = int(seconds)
        header.stamp.nanosec = int(round((seconds - header.stamp.sec) * 1e9))

    def _joint_state(self, observation: Any) -> Any:
        message = self._joint_state_type()
        message.name = list(observation.joint_state.name)
        message.position = observation.joint_state.position.tolist()
        message.velocity = observation.joint_state.velocity.tolist()
        message.effort = observation.joint_state.effort.tolist()
        message.header.frame_id = "base_link"
        self._stamp(message.header, observation.sim_time)
        return message

    def _odom(self, observation: Any) -> Any:
        pose = observation.ee_pose.pose
        message = self._odom_type()
        message.header.frame_id = "odom"
        message.child_frame_id = "base_link"
        self._stamp(message.header, observation.sim_time)
        message.pose.pose.position.x = pose.position.x
        message.pose.pose.position.y = pose.position.y
        message.pose.pose.position.z = pose.position.z
        message.pose.pose.orientation.x = pose.orientation.x
        message.pose.pose.orientation.y = pose.orientation.y
        message.pose.pose.orientation.z = pose.orientation.z
        message.pose.pose.orientation.w = pose.orientation.w
        message.twist.twist.linear.x = self._command.linear.x
        message.twist.twist.linear.y = self._command.linear.y
        message.twist.twist.angular.z = self._command.angular.z
        return message

    def _tf(self, observation: Any) -> Any:
        pose = observation.ee_pose.pose
        transform = self._transform_type()
        transform.header.frame_id = "odom"
        transform.child_frame_id = "base_link"
        self._stamp(transform.header, observation.sim_time)
        transform.transform.translation.x = pose.position.x
        transform.transform.translation.y = pose.position.y
        transform.transform.translation.z = pose.position.z
        transform.transform.rotation.x = pose.orientation.x
        transform.transform.rotation.y = pose.orientation.y
        transform.transform.rotation.z = pose.orientation.z
        transform.transform.rotation.w = pose.orientation.w
        message = self._tf_message_type()
        message.transforms = [transform]
        return message

    def publish_observation(self, observation: Any) -> None:
        self._joint_state_publisher.publish(self._joint_state(observation))
        self._odom_publisher.publish(self._odom(observation))
        self._tf_publisher.publish(self._tf(observation))

    def reset(self, seed: int | None = None) -> Any:
        observation = self.simulation.reset(seed=seed)
        self.publish_observation(observation)
        return observation

    def run(self, *, seed: int | None = None, max_ticks: int | None = None) -> int:
        from rclpy.executors import SingleThreadedExecutor

        observation = self.reset(seed=seed)
        period = 1.0 / float(self.simulation.cfg.control_hz)
        deadline = time.monotonic()
        ticks = 0
        executor = SingleThreadedExecutor(context=self.node.context)
        executor.add_node(self.node)
        try:
            while self.node.context.ok() and (max_ticks is None or ticks < max_ticks):
                executor.spin_once(timeout_sec=0.0)
                observation, _, terminated, truncated, _ = self.simulation.step(
                    Action(ee_twist=self._command)
                )
                self.publish_observation(observation)
                ticks += 1
                if terminated or truncated:
                    break
                deadline += period
                delay = deadline - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    deadline = time.monotonic()
        finally:
            executor.remove_node(self.node)
        return ticks

    def close(self) -> None:
        self.simulation.close()
        self.node.destroy_subscription(self._cmd_subscription)
        self.node.destroy_publisher(self._joint_state_publisher)
        self.node.destroy_publisher(self._odom_publisher)
        self.node.destroy_publisher(self._tf_publisher)


__all__ = ["TurtleBot4ROS2Node"]