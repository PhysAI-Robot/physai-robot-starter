"""ALOHA 2 follower-arm training and joint-order contracts.

One ALOHA 2 follower arm is a ViperX 300 S: six arm joints and a parallel gripper whose single
actuator drives the left finger (the right finger mirrors it). The gripper's joint is prismatic,
so its value in `observation.state` and in `action` is a finger position in metres, not radians.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ...contracts import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    Action,
    ActionSpec,
    CameraSpec,
    ObservationSpec,
    TensorSpec,
)
from ..base import RobotTrainingContract

ARM_JOINT_NAMES: tuple[str, ...] = (
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
)
# The name the robot exposes. The MJCF joint is `left_finger` and its actuator is `gripper`.
GRIPPER_JOINT_NAME = "gripper"
# The finger position (m) used when a recorder has no gripper command to encode.
GRIPPER_OPEN_M = 0.057
ALL_JOINT_NAMES: tuple[str, ...] = ARM_JOINT_NAMES + (GRIPPER_JOINT_NAME,)
VIPERX300S_ACTION_SCHEMA = "viperx300s.joint_position.v1"
VIPERX300S_OBSERVATION_SCHEMA = "viperx300s.observation.v1"
STATE_UNITS = "rad (gripper: m)"


def viperx300s_action_values(action: Action, *, gripper_joint: float) -> np.ndarray:
    """Return absolute ViperX 300 S targets in the canonical joint order."""
    if action.joint_position is None:
        raise ValueError("ViperX 300 S action requires joint-position targets")
    if action.joint_position.size != len(ARM_JOINT_NAMES):
        raise ValueError(
            f"ViperX 300 S expects {len(ARM_JOINT_NAMES)} arm targets, "
            f"got {action.joint_position.size}"
        )
    if action.joint_names is not None and action.joint_names != ARM_JOINT_NAMES:
        raise ValueError(
            f"ViperX 300 S expects joint order {ARM_JOINT_NAMES}, got {action.joint_names}"
        )
    values = np.concatenate([action.joint_position, [gripper_joint]])
    if not np.isfinite(values).all():
        raise ValueError("ViperX 300 S action contains non-finite values")
    return values


def viperx300s_action_encoder(
    action: Action, *, gripper_joint: float
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Encode an action for the recorder without leaking ViperX 300 S rules into data."""
    return viperx300s_action_values(
        action, gripper_joint=gripper_joint
    ), ALL_JOINT_NAMES


def viperx300s_action_spec() -> ActionSpec:
    """Describe the stable seven-value ViperX 300 S training action layout."""
    return ActionSpec(
        fields=(
            TensorSpec(
                name="action",
                shape=(len(ALL_JOINT_NAMES),),
                dtype="float32",
                units=STATE_UNITS,
            ),
        ),
        metadata={
            "schema": VIPERX300S_ACTION_SCHEMA,
            "mode": "joint_position",
            "absolute": True,
            "names": list(ALL_JOINT_NAMES),
            "joint_names": list(ALL_JOINT_NAMES),
        },
    )


def viperx300s_action_schema() -> dict[str, Any]:
    """Return the serialized canonical ViperX 300 S action feature metadata."""
    spec = viperx300s_action_spec()
    return {
        **spec.metadata,
        "names": list(ALL_JOINT_NAMES),
        "dtype": "float32",
        "shape": [len(ALL_JOINT_NAMES)],
        "units": STATE_UNITS,
    }


def viperx300s_observation_spec(
    *,
    camera_config: Mapping[str, Mapping[str, Any]] | None = None,
) -> ObservationSpec:
    """Describe the canonical ViperX 300 S state and camera observation layout."""
    camera_config = camera_config or {"front": {}, "wrist": {}}
    cameras = []
    for name, config in camera_config.items():
        cameras.append(
            CameraSpec(
                name=name,
                shape=(
                    int(config.get("height", CAMERA_HEIGHT)),
                    int(config.get("width", CAMERA_WIDTH)),
                    3,
                ),
                dtype=str(config.get("dtype", "uint8")),
                encoding=str(config.get("encoding", "rgb8")),
                frame_id=str(config.get("frame_id", "")),
            )
        )
    return ObservationSpec(
        fields=(
            TensorSpec(
                name="observation.state",
                shape=(len(ALL_JOINT_NAMES),),
                dtype="float32",
                units=STATE_UNITS,
            ),
        ),
        cameras=tuple(cameras),
        metadata={
            "schema": VIPERX300S_OBSERVATION_SCHEMA,
            "joint_names": list(ALL_JOINT_NAMES),
        },
    )


def viperx300s_observation_schema(
    *,
    camera_config: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return the serialized observation feature map used by dataset metadata."""
    spec = viperx300s_observation_spec(camera_config=camera_config)
    schema = {
        "observation.state": {
            **spec.fields[0].to_dict(),
            "names": list(ALL_JOINT_NAMES),
        }
    }
    schema.update(
        {
            f"observation.images.{camera.name}": camera.to_dict()
            for camera in spec.cameras
        }
    )
    return schema


def viperx300s_training_contract(
    *, camera_config: Mapping[str, Mapping[str, Any]] | None = None
) -> RobotTrainingContract:
    """Return the complete ViperX 300 S contract consumed by training adapters."""
    return RobotTrainingContract(
        observation_spec=viperx300s_observation_spec(camera_config=camera_config),
        action_spec=viperx300s_action_spec(),
        action_encoder=lambda action, gripper_joint: viperx300s_action_encoder(
            action,
            gripper_joint=GRIPPER_OPEN_M if gripper_joint is None else gripper_joint,
        ),
        action_schema=viperx300s_action_schema(),
        observation_schema=viperx300s_observation_schema(camera_config=camera_config),
    )


__all__ = [
    "ALL_JOINT_NAMES",
    "ARM_JOINT_NAMES",
    "GRIPPER_JOINT_NAME",
    "GRIPPER_OPEN_M",
    "VIPERX300S_ACTION_SCHEMA",
    "VIPERX300S_OBSERVATION_SCHEMA",
    "viperx300s_action_encoder",
    "viperx300s_action_schema",
    "viperx300s_action_spec",
    "viperx300s_action_values",
    "viperx300s_observation_schema",
    "viperx300s_observation_spec",
    "viperx300s_training_contract",
]
