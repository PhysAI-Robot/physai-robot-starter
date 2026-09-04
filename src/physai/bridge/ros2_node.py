"""Executable ROS2 boundary for the SO-101 MuJoCo simulation."""

from __future__ import annotations

import math
import time
from typing import Any

import mujoco
import numpy as np

from ..robots.so101 import EnvConfig, SO101Env
from .messages import ROS2MessageCodec
from .mujoco_ros_bridge import MuJoCoROSBridge, RclpyTransport


def _quaternion_from_rotation(rotation: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / scale
        x = 0.25 * scale
        y = (rotation[0, 1] + rotation[1, 0]) / scale
        z = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / scale
        x = (rotation[0, 1] + rotation[1, 0]) / scale
        y = 0.25 * scale
        z = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / scale
        x = (rotation[0, 2] + rotation[2, 0]) / scale
        y = (rotation[1, 2] + rotation[2, 1]) / scale
        z = 0.25 * scale
    return float(x), float(y), float(z), float(w)


class SO101ROS2Node:
    """Run the SO-101 MuJoCo port through actual ROS2 publishers/subscribers."""

    def __init__(self, node: Any, config: EnvConfig | None = None) -> None:
        try:
            from control_msgs.msg import GripperCommand
            from geometry_msgs.msg import TransformStamped
            from sensor_msgs.msg import CameraInfo, Image, JointState
            from tf2_msgs.msg import TFMessage
            from trajectory_msgs.msg import JointTrajectory
        except ImportError as exc:
            raise RuntimeError(
                "SO-101 ROS2 simulation needs ros-jazzy-control-msgs; "
                "install it or use the provided Docker image"
            ) from exc

        self.node = node
        self.simulation = SO101Env(config or EnvConfig(render=True))
        transport = RclpyTransport(node, {
            "/arm_controller/joint_trajectory": JointTrajectory,
            "/gripper_controller/gripper_cmd": GripperCommand,
        })
        codec = ROS2MessageCodec(JointState, Image)
        self.bridge = MuJoCoROSBridge(self.simulation, transport, codec=codec)
        self._camera_info_type = CameraInfo
        self._transform_type = TransformStamped
        self._tf_message_type = TFMessage
        self._camera_info_publishers = {
            name: node.create_publisher(CameraInfo, f"/camera/{name}/camera_info", 10)
            for name in self.simulation.cfg.cameras
        }
        self._tf_publisher = node.create_publisher(TFMessage, "/tf", 10)

    def _camera_info(self, name: str, observation: Any) -> Any:
        image = observation.images[name]
        message = self._camera_info_type()
        message.header.frame_id = image.header.frame_id
        _copy_ros_stamp(message.header, image.header.stamp)
        message.width = image.width
        message.height = image.height
        camera_id = mujoco.mj_name2id(
            self.simulation.model, mujoco.mjtObj.mjOBJ_CAMERA, name
        )
        fovy = math.radians(float(self.simulation.model.cam_fovy[camera_id]))
        focal = image.height / (2.0 * math.tan(fovy / 2.0))
        center_x = (image.width - 1) / 2.0
        center_y = (image.height - 1) / 2.0
        message.distortion_model = "plumb_bob"
        message.d = [0.0] * 5
        message.k = [focal, 0.0, center_x, 0.0, focal, center_y, 0.0, 0.0, 1.0]
        message.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        message.p = [focal, 0.0, center_x, 0.0, 0.0, focal, center_y, 0.0, 0.0, 0.0, 1.0, 0.0]
        return message

    def _tf_message(self, observation: Any) -> Any:
        model = self.simulation.model
        data = self.simulation.data
        frames = ("base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper")
        transforms = []
        world_positions = {"world": np.zeros(3)}
        world_rotations = {"world": np.eye(3)}
        for frame in frames:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, frame)
            world_positions[frame] = data.xpos[body_id].copy()
            world_rotations[frame] = data.xmat[body_id].reshape(3, 3).copy()
        parent = "world"
        for child in frames:
            transforms.append(self._transform(parent, child, world_positions, world_rotations, observation))
            parent = child
        world_positions["gripper_frame"] = world_positions["gripper"]
        world_rotations["gripper_frame"] = world_rotations["gripper"]
        transforms.append(self._transform(
            "gripper", "gripper_frame", world_positions, world_rotations, observation
        ))
        for camera_name, frame_name, parent in (
            ("front", "camera_front", "world"),
            ("wrist", "camera_wrist", "gripper_frame"),
        ):
            camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            world_positions[frame_name] = data.cam_xpos[camera_id].copy()
            world_rotations[frame_name] = data.cam_xmat[camera_id].reshape(3, 3).copy()
            transforms.append(self._transform(parent, frame_name, world_positions, world_rotations, observation))
        message = self._tf_message_type()
        message.transforms = transforms
        return message

    def _transform(self, parent: str, child: str, positions: dict[str, np.ndarray], rotations: dict[str, np.ndarray], observation: Any) -> Any:
        parent_rotation = rotations[parent]
        translation = parent_rotation.T @ (positions[child] - positions[parent])
        rotation = parent_rotation.T @ rotations[child]
        transform = self._transform_type()
        transform.header.frame_id = parent
        transform.child_frame_id = child
        _copy_ros_stamp(transform.header, observation.joint_state.header.stamp)
        transform.transform.translation.x = float(translation[0])
        transform.transform.translation.y = float(translation[1])
        transform.transform.translation.z = float(translation[2])
        x, y, z, w = _quaternion_from_rotation(rotation)
        transform.transform.rotation.x = x
        transform.transform.rotation.y = y
        transform.transform.rotation.z = z
        transform.transform.rotation.w = w
        return transform

    def publish_extras(self, observation: Any) -> None:
        for name, publisher in self._camera_info_publishers.items():
            if name in observation.images:
                publisher.publish(self._camera_info(name, observation))
        self._tf_publisher.publish(self._tf_message(observation))

    def reset(self, seed: int | None = None) -> Any:
        observation = self.bridge.reset(seed=seed)
        self.publish_extras(observation)
        return observation

    def run(self, *, seed: int | None = None, max_ticks: int | None = None) -> int:
        import rclpy

        observation = self.reset(seed=seed)
        period = 1.0 / float(self.simulation.cfg.control_hz)
        deadline = time.monotonic()
        ticks = 0
        while rclpy.ok() and (max_ticks is None or ticks < max_ticks):
            rclpy.spin_once(self.node, timeout_sec=0.0)
            observation, _, terminated, truncated, _ = self.bridge.tick()
            self.publish_extras(observation)
            ticks += 1
            if terminated or truncated:
                break
            deadline += period
            delay = deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                deadline = time.monotonic()
        return ticks

    def close(self) -> None:
        self.bridge.close()


def _copy_ros_stamp(header: Any, stamp: float) -> None:
    seconds = max(0.0, float(stamp))
    header.stamp.sec = int(seconds)
    header.stamp.nanosec = int(round((seconds - header.stamp.sec) * 1e9))


__all__ = ["SO101ROS2Node"]