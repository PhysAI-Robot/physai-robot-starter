"""Conversions between internal contracts and ROS2-shaped messages.

The default codec keeps the Phase 0 dataclasses intact. A ROS2 node can inject
a codec that constructs real ``sensor_msgs`` and ``trajectory_msgs`` values
without making the core package depend on ``rclpy``.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from ..contracts import Action, GripperCommand, ImageFrame, JointState


class MessageCodec(Protocol):
    """Encode observations and decode commands at the transport boundary."""

    def encode_joint_state(self, value: JointState) -> Any: ...

    def encode_image(self, value: ImageFrame) -> Any: ...

    def decode_joint_trajectory(self, value: Any) -> Action: ...

    def decode_gripper_command(self, value: Any) -> GripperCommand: ...


class ContractMessageCodec:
    """Codec for Phase 0 contract objects and ROS2-shaped input objects."""

    def encode_joint_state(self, value: JointState) -> JointState:
        return value

    def encode_image(self, value: ImageFrame) -> ImageFrame:
        return value

    def decode_joint_trajectory(self, value: Any) -> Action:
        if isinstance(value, Action):
            return value
        points = getattr(value, "points", ())
        if not points:
            raise ValueError("joint trajectory must contain at least one point")
        point = points[0]
        joint_names = tuple(getattr(value, "joint_names", ()))
        if not joint_names:
            raise ValueError("joint trajectory must contain joint_names")
        positions = np.asarray(getattr(point, "positions", ()), dtype=np.float64)
        if positions.size != len(joint_names):
            raise ValueError("joint trajectory names and positions have different sizes")
        return Action(
            joint_position=positions,
            joint_names=joint_names,
            stamp=_header_stamp(getattr(value, "header", None)),
        )

    def decode_gripper_command(self, value: Any) -> GripperCommand:
        if isinstance(value, GripperCommand):
            return value
        command = getattr(value, "command", value)
        return GripperCommand(
            position=float(getattr(command, "position")),
            max_effort=float(getattr(command, "max_effort", 0.0)),
        )


class ROS2MessageCodec(ContractMessageCodec):
    """Codec that fills real ROS2 message instances from injected factories."""

    def __init__(self, joint_state_factory: Any, image_factory: Any) -> None:
        self._joint_state_factory = joint_state_factory
        self._image_factory = image_factory

    def encode_joint_state(self, value: JointState) -> Any:
        message = self._joint_state_factory()
        message.name = list(value.name)
        message.position = value.position.tolist()
        message.velocity = value.velocity.tolist()
        message.effort = value.effort.tolist()
        _copy_header(message, value.header)
        return message

    def encode_image(self, value: ImageFrame) -> Any:
        image = np.asarray(value.data, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("ROS2 rgb8 images must have shape (height, width, 3)")
        message = self._image_factory()
        message.height = int(image.shape[0])
        message.width = int(image.shape[1])
        message.encoding = value.encoding
        message.is_bigendian = 0
        message.step = int(image.shape[1] * 3)
        message.data = np.ascontiguousarray(image).tobytes()
        _copy_header(message, value.header)
        return message


def _header_stamp(header: Any) -> float | None:
    if header is None or not hasattr(header, "stamp"):
        return None
    stamp = header.stamp
    if isinstance(stamp, (int, float)):
        return float(stamp)
    seconds = float(getattr(stamp, "sec", 0))
    nanoseconds = float(getattr(stamp, "nanosec", 0))
    if seconds == 0.0 and nanoseconds == 0.0:
        return None
    return seconds + nanoseconds * 1e-9


def _copy_header(message: Any, source: Any) -> None:
    header = getattr(message, "header", None)
    if header is None:
        return
    header.frame_id = source.frame_id
    stamp = getattr(header, "stamp", None)
    if hasattr(stamp, "sec") and hasattr(stamp, "nanosec"):
        seconds = max(0.0, float(source.stamp))
        stamp.sec = int(seconds)
        stamp.nanosec = int(round((seconds - stamp.sec) * 1e9))
    else:
        header.stamp = float(source.stamp)


__all__ = ["ContractMessageCodec", "MessageCodec", "ROS2MessageCodec"]