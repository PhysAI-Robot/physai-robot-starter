"""ROS2 node adapter for the TurtleBot4 MuJoCo backend."""

from __future__ import annotations

import math
import time
from typing import Any

from ...contracts import Action, Twist, Vector3
from .env import TurtleBot4Config, TurtleBot4Env


class TurtleBot4ROS2Node:
    """Run TurtleBot4 through real ROS2 command and state topics."""

    def __init__(
        self,
        node: Any,
        config: TurtleBot4Config | None = None,
        scenario: str = "open_space",
    ) -> None:
        from geometry_msgs.msg import TransformStamped
        from geometry_msgs.msg import Twist as ROSTwist
        from nav_msgs.msg import Odometry
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import JointState, LaserScan
        from tf2_msgs.msg import TFMessage

        self.node = node
        if config is not None and scenario != "open_space":
            raise ValueError("pass either config or a non-default TurtleBot4 scenario")
        if config is None and scenario == "obstacle_course":
            config = TurtleBot4Config(
                obstacles=((0.0, -0.5, 0.4, 0.1, 0.4),),
            )
        elif config is None and scenario != "open_space":
            raise ValueError(f"unknown TurtleBot4 scenario: {scenario}")
        self.simulation = TurtleBot4Env(config)
        self._odom_type = Odometry
        self._joint_state_type = JointState
        self._tf_message_type = TFMessage
        self._transform_type = TransformStamped
        self._scan_type = LaserScan
        self._clock_type = Clock
        self._command = Twist()
        self._cmd_subscription = node.create_subscription(
            ROSTwist, "/cmd_vel", self._receive_twist, 10
        )
        self._joint_state_publisher = node.create_publisher(
            JointState, "/joint_states", 10
        )
        self._odom_publisher = node.create_publisher(Odometry, "/odom", 10)
        self._tf_publisher = node.create_publisher(TFMessage, "/tf", 10)
        self._scan_publisher = node.create_publisher(LaserScan, "/scan", 10)
        self._clock_publisher = node.create_publisher(Clock, "/clock", 10)

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

    def _ros_orientation(self, pose: Any) -> tuple[float, float, float, float]:
        yaw = 2.0 * math.atan2(float(pose.orientation.z), float(pose.orientation.w))
        return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)

    def _ros_position(self, position: Any) -> tuple[float, float, float]:
        return -float(position.y), float(position.x), float(position.z)

    def _stamp(self, header: Any, value: float) -> None:
        seconds = max(0.0, float(value))
        stamp = header.stamp if hasattr(header, "stamp") else header
        stamp.sec = int(seconds)
        stamp.nanosec = int(round((seconds - stamp.sec) * 1e9))

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
        position = self._ros_position(pose.position)
        message.pose.pose.position.x = position[0]
        message.pose.pose.position.y = position[1]
        message.pose.pose.position.z = position[2]
        orientation = self._ros_orientation(pose)
        message.pose.pose.orientation.x = orientation[0]
        message.pose.pose.orientation.y = orientation[1]
        message.pose.pose.orientation.z = orientation[2]
        message.pose.pose.orientation.w = orientation[3]
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
        position = self._ros_position(pose.position)
        transform.transform.translation.x = position[0]
        transform.transform.translation.y = position[1]
        transform.transform.translation.z = position[2]
        orientation = self._ros_orientation(pose)
        transform.transform.rotation.x = orientation[0]
        transform.transform.rotation.y = orientation[1]
        transform.transform.rotation.z = orientation[2]
        transform.transform.rotation.w = orientation[3]
        message = self._tf_message_type()
        message.transforms = [transform]
        return message

    def _scan(self, observation: Any) -> Any:
        message = self._scan_type()
        cfg = self.simulation.cfg
        message.header.frame_id = "base_link"
        self._stamp(message.header, observation.sim_time)
        message.angle_min = float(cfg.lidar_angle_min)
        message.angle_max = float(cfg.lidar_angle_max)
        message.angle_increment = 2.0 * math.pi / float(cfg.lidar_samples)
        message.time_increment = 0.0
        message.scan_time = 1.0 / float(cfg.control_hz)
        message.range_min = float(cfg.lidar_range_min)
        message.range_max = float(cfg.lidar_range_max)
        message.ranges = self.simulation.lidar_ranges().tolist()
        return message

    def _clock(self, observation: Any) -> Any:
        message = self._clock_type()
        self._stamp(message.clock, observation.sim_time)
        return message

    def publish_observation(self, observation: Any) -> None:
        self._clock_publisher.publish(self._clock(observation))
        self._joint_state_publisher.publish(self._joint_state(observation))
        self._odom_publisher.publish(self._odom(observation))
        self._tf_publisher.publish(self._tf(observation))
        self._scan_publisher.publish(self._scan(observation))

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
        self.node.destroy_publisher(self._scan_publisher)
        self.node.destroy_publisher(self._clock_publisher)


__all__ = ["TurtleBot4ROS2Node"]