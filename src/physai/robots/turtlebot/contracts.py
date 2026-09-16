"""TurtleBot4 observation and action contracts for training adapters."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...contracts import (
    Action,
    ActionSpec,
    CameraSpec,
    ObservationSpec,
    TensorSpec,
    Twist,
    Vector3,
)
from ..base import RobotTrainingContract

TURTLEBOT4_JOINT_NAMES = ("left_wheel", "right_wheel")
TURTLEBOT4_ACTION_SCHEMA = "turtlebot4.twist.v1"
TURTLEBOT4_OBSERVATION_SCHEMA = "turtlebot4.observation.v1"
_TWIST_NAMES = (
    "linear_x",
    "linear_y",
    "linear_z",
    "angular_x",
    "angular_y",
    "angular_z",
)


def turtlebot4_action_encoder(
    action: Action, _gripper_joint: float | None = None
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Encode a base twist into the canonical six-value ROS order."""
    if action.ee_twist is None:
        raise ValueError("TurtleBot4 training actions require a twist")
    return action.ee_twist.as_array(), _TWIST_NAMES


def turtlebot4_action_spec() -> ActionSpec:
    """Describe the canonical TurtleBot4 twist action layout."""
    return ActionSpec(
        fields=(
            TensorSpec(
                name="action",
                shape=(6,),
                dtype="float32",
                units="m/s,rad/s",
            ),
        ),
        metadata={
            "schema": TURTLEBOT4_ACTION_SCHEMA,
            "mode": "twist",
            "frame": "base",
            "names": list(_TWIST_NAMES),
        },
    )


def turtlebot4_action_schema() -> dict[str, Any]:
    """Return serialized TurtleBot4 action metadata for dataset manifests."""
    spec = turtlebot4_action_spec()
    return {
        **spec.metadata,
        "dtype": "float32",
        "shape": [6],
        "units": "m/s,rad/s",
    }


def turtlebot4_action_decoder(values: np.ndarray) -> Action:
    """Decode the canonical twist vector into a shared ``Action`` value."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size != 6:
        raise ValueError(f"TurtleBot4 expects six twist values, got {values.size}")
    return Action(
        ee_twist=Twist(
            linear=Vector3(*values[:3]),
            angular=Vector3(*values[3:]),
            frame_id="base",
        )
    )


def turtlebot4_observation_spec(
    *, camera_config: dict[str, dict[str, Any]] | None = None
) -> ObservationSpec:
    """Describe wheel state and the TurtleBot4 base camera observation."""
    config = (camera_config or {}).get("free", {})
    return ObservationSpec(
        fields=(
            TensorSpec(
                name="observation.state",
                shape=(len(TURTLEBOT4_JOINT_NAMES),),
                dtype="float32",
                units="rad",
            ),
        ),
        cameras=(
            CameraSpec(
                name="free",
                shape=(
                    int(config.get("height", 480)),
                    int(config.get("width", 640)),
                    3,
                ),
                dtype=str(config.get("dtype", "uint8")),
                encoding=str(config.get("encoding", "rgb8")),
                frame_id=str(config.get("frame_id", "base_link")),
            ),
        ),
        metadata={
            "schema": TURTLEBOT4_OBSERVATION_SCHEMA,
            "joint_names": list(TURTLEBOT4_JOINT_NAMES),
        },
    )


def turtlebot4_observation_schema(
    *, camera_config: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Return serialized TurtleBot4 observation metadata for dataset manifests."""
    spec = turtlebot4_observation_spec(camera_config=camera_config)
    schema = {
        "observation.state": {
            **spec.fields[0].to_dict(),
            "names": list(TURTLEBOT4_JOINT_NAMES),
        }
    }
    schema.update(
        {
            f"observation.images.{camera.name}": camera.to_dict()
            for camera in spec.cameras
        }
    )
    return schema


def turtlebot4_training_contract(
    *, camera_config: dict[str, dict[str, Any]] | None = None
) -> RobotTrainingContract:
    """Return the complete TurtleBot4 contract for training adapters."""
    return RobotTrainingContract(
        observation_spec=turtlebot4_observation_spec(camera_config=camera_config),
        action_spec=turtlebot4_action_spec(),
        action_encoder=turtlebot4_action_encoder,
        action_decoder=turtlebot4_action_decoder,
        action_schema=turtlebot4_action_schema(),
        observation_schema=turtlebot4_observation_schema(camera_config=camera_config),
    )


__all__ = [
    "TURTLEBOT4_ACTION_SCHEMA",
    "TURTLEBOT4_JOINT_NAMES",
    "TURTLEBOT4_OBSERVATION_SCHEMA",
    "turtlebot4_action_decoder",
    "turtlebot4_action_encoder",
    "turtlebot4_action_schema",
    "turtlebot4_action_spec",
    "turtlebot4_observation_schema",
    "turtlebot4_observation_spec",
    "turtlebot4_training_contract",
]
