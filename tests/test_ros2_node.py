"""Acceptance coverage for the executable ROS2 SO-101 boundary."""

from __future__ import annotations

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("control_msgs.msg")
pytest.importorskip("sensor_msgs.msg")
pytest.importorskip("tf2_msgs.msg")
pytest.importorskip("trajectory_msgs.msg")
pytest.importorskip("geometry_msgs.msg")


def test_real_ros2_trajectory_and_gripper_move_mujoco():
    from control_msgs.msg import GripperCommand
    from sensor_msgs.msg import CameraInfo, JointState
    from tf2_msgs.msg import TFMessage
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    from physai.bridge import SO101ROS2Node
    from physai.robots.so101 import EnvConfig

    rclpy.init(args=[])
    node = rclpy.create_node("test_so101_ros2_node")
    driver = SO101ROS2Node(node, EnvConfig(render=True, cameras=("front",)))
    published_joint_states = []
    published_camera_info = []
    published_tf = []
    node.create_subscription(
        JointState, "/joint_states", published_joint_states.append, 10
    )
    node.create_subscription(
        CameraInfo, "/camera/front/camera_info", published_camera_info.append, 10
    )
    node.create_subscription(TFMessage, "/tf", published_tf.append, 10)
    trajectory_publisher = node.create_publisher(
        JointTrajectory, "/arm_controller/joint_trajectory", 10
    )
    gripper_publisher = node.create_publisher(
        GripperCommand, "/gripper_controller/gripper_cmd", 10
    )
    try:
        initial = driver.reset(seed=42).joint_state.position.copy()
        trajectory = JointTrajectory()
        trajectory.joint_names = [
            "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll",
        ]
        point = JointTrajectoryPoint()
        point.positions = [0.2, -0.8, 1.0, 0.5, 0.1]
        trajectory.points = [point]
        gripper = GripperCommand(position=0.2, max_effort=2.0)
        trajectory_publisher.publish(trajectory)
        gripper_publisher.publish(gripper)

        driver.run(max_ticks=30)
        rclpy.spin_once(node, timeout_sec=0.1)
        final = driver.bridge.observation.joint_state.position

        np.testing.assert_allclose(final[:5], point.positions, atol=0.05)
        assert not np.allclose(final, initial)
        assert published_joint_states
        assert published_camera_info
        assert published_camera_info[-1].width > 0
        assert published_tf
        assert "gripper_frame" in {
            transform.child_frame_id for transform in published_tf[-1].transforms
        }
    finally:
        driver.close()
        node.destroy_node()
        rclpy.shutdown()