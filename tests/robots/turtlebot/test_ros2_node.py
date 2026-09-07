"""Acceptance coverage for the executable ROS2 TurtleBot4 boundary."""

from __future__ import annotations

import pytest


rclpy = pytest.importorskip("rclpy")
pytest.importorskip("geometry_msgs.msg")
pytest.importorskip("nav_msgs.msg")
pytest.importorskip("sensor_msgs.msg")
pytest.importorskip("tf2_msgs.msg")


def test_real_ros2_cmd_vel_publishes_turtlebot_state_and_tf():
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import JointState
    from tf2_msgs.msg import TFMessage

    from physai.robots.turtlebot import TurtleBot4Config, TurtleBot4ROS2Node

    rclpy.init(args=[])
    node = rclpy.create_node("test_turtlebot4_ros2_node")
    driver = TurtleBot4ROS2Node(node, TurtleBot4Config(render=False))
    published_joint_states = []
    published_odom = []
    published_tf = []
    node.create_subscription(JointState, "/joint_states", published_joint_states.append, 10)
    node.create_subscription(Odometry, "/odom", published_odom.append, 10)
    node.create_subscription(TFMessage, "/tf", published_tf.append, 10)
    cmd_publisher = node.create_publisher(Twist, "/cmd_vel", 10)
    try:
        initial = driver.reset(seed=4).ee_pose.pose.position.as_array().copy()
        command = Twist()
        command.linear.x = 0.4
        command.angular.z = 0.2
        cmd_publisher.publish(command)
        driver.run(max_ticks=20)
        for _ in range(3):
            rclpy.spin_once(node, timeout_sec=0.1)

        final_observation = driver.simulation.observe()
        assert final_observation.ee_pose is not None
        final = final_observation.ee_pose.pose.position.as_array()
        assert final[0] > initial[0] + 0.1
        assert published_joint_states
        assert published_odom
        assert published_tf
        assert published_joint_states[-1].name == ["left_wheel", "right_wheel"]
        assert published_odom[-1].child_frame_id == "base_link"
        assert [(item.header.frame_id, item.child_frame_id) for item in published_tf[-1].transforms] == [
            ("odom", "base_link")
        ]
    finally:
        driver.close()
        node.destroy_node()
        rclpy.shutdown()