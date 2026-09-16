"""SO-101-specific training and joint-order contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ...contracts import (
    Action,
    ActionSpec,
    CameraSpec,
    ObservationSpec,
    TensorSpec,
)
from ..base import RobotTrainingContract

ARM_JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
GRIPPER_JOINT_NAME = "gripper"
ALL_JOINT_NAMES: tuple[str, ...] = ARM_JOINT_NAMES + (GRIPPER_JOINT_NAME,)
SO101_ACTION_SCHEMA = "so101.joint_position.v1"
SO101_OBSERVATION_SCHEMA = "so101.observation.v1"


def so101_action_values(action: Action, *, gripper_joint: float) -> np.ndarray:
    """Return absolute SO-101 targets in the canonical joint order."""
    if action.joint_position is None:
        raise ValueError("SO-101 action requires joint-position targets")
    if action.joint_position.size != len(ARM_JOINT_NAMES):
        raise ValueError(
            f"SO-101 expects {len(ARM_JOINT_NAMES)} arm targets, "
            f"got {action.joint_position.size}"
        )
    if action.joint_names is not None and action.joint_names != ARM_JOINT_NAMES:
        raise ValueError(
            f"SO-101 expects joint order {ARM_JOINT_NAMES}, got {action.joint_names}"
        )
    values = np.concatenate([action.joint_position, [gripper_joint]])
    if not np.isfinite(values).all():
        raise ValueError("SO-101 action contains non-finite values")
    return values


def so101_action_encoder(
    action: Action, *, gripper_joint: float
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Encode an action for the recorder without leaking SO-101 rules into data."""
    return so101_action_values(action, gripper_joint=gripper_joint), ALL_JOINT_NAMES


def so101_action_schema() -> dict[str, Any]:
    """Return the serialized canonical SO-101 action feature metadata."""
    spec = so101_action_spec()
    return {
        **spec.metadata,
        "names": list(ALL_JOINT_NAMES),
        "dtype": "float32",
        "shape": [len(ALL_JOINT_NAMES)],
        "units": "rad",
    }


def so101_action_spec() -> ActionSpec:
    """Describe the stable six-value SO-101 training action layout."""
    return ActionSpec(
        fields=(
            TensorSpec(
                name="action",
                shape=(len(ALL_JOINT_NAMES),),
                dtype="float32",
                units="rad",
            ),
        ),
        metadata={
            "schema": SO101_ACTION_SCHEMA,
            "mode": "joint_position",
            "absolute": True,
            "names": list(ALL_JOINT_NAMES),
            "joint_names": list(ALL_JOINT_NAMES),
        },
    )


def so101_observation_spec(
    *,
    camera_config: Mapping[str, Mapping[str, Any]] | None = None,
) -> ObservationSpec:
    """Describe the canonical SO-101 state and camera observation layout."""
    camera_config = camera_config or {"front": {}, "wrist": {}}
    cameras = []
    for name, config in camera_config.items():
        cameras.append(
            CameraSpec(
                name=name,
                shape=(
                    int(config.get("height", 224)),
                    int(config.get("width", 224)),
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
                units="rad",
            ),
        ),
        cameras=tuple(cameras),
        metadata={
            "schema": SO101_OBSERVATION_SCHEMA,
            "joint_names": list(ALL_JOINT_NAMES),
        },
    )


def so101_observation_schema(
    *,
    camera_config: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return the serialized observation feature map used by dataset metadata."""
    spec = so101_observation_spec(camera_config=camera_config)
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


def so101_training_contract(
    *, camera_config: Mapping[str, Mapping[str, Any]] | None = None
) -> RobotTrainingContract:
    """Return the complete SO-101 contract consumed by training adapters."""
    return RobotTrainingContract(
        observation_spec=so101_observation_spec(camera_config=camera_config),
        action_spec=so101_action_spec(),
        action_encoder=lambda action, gripper_joint: so101_action_encoder(
            action, gripper_joint=0.0 if gripper_joint is None else gripper_joint
        ),
        action_schema=so101_action_schema(),
        observation_schema=so101_observation_schema(camera_config=camera_config),
    )


__all__ = [
    "ALL_JOINT_NAMES",
    "ARM_JOINT_NAMES",
    "GRIPPER_JOINT_NAME",
    "SO101_ACTION_SCHEMA",
    "SO101_OBSERVATION_SCHEMA",
    "so101_action_encoder",
    "so101_action_schema",
    "so101_action_spec",
    "so101_action_values",
    "so101_observation_schema",
    "so101_observation_spec",
    "so101_training_contract",
]
