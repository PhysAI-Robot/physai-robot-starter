"""Per-step signals a run can attach to a recorded episode, rosbag style."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from ..contracts import Observation


def _value(value: Any) -> Any:
    """NaN for a missing value, float for a bool/number, text and arrays as is."""
    if value is None:
        return np.nan
    if isinstance(value, str | np.ndarray):
        return value
    if isinstance(value, bool | int | float | np.number | np.bool_):
        return float(value)
    return np.asarray(value, dtype=np.float64)


def collect_extras(
    robot: Any, policy: Any, observation: Observation, info: dict | None
) -> dict[str, Any]:
    """Everything the observation, robot, policy and last step info expose.

    Duck-typed so any robot or policy contributes what it has: joint
    velocity/effort, end-effector pose, `robot.gripper_contact_force()`, the
    fields of `policy.metrics` (visual servo: phase, errors, detection) and
    the scalar entries of the step `info`.
    """
    extras: dict[str, Any] = {}
    joints = observation.joint_state
    if joints.velocity is not None:
        extras["joint_velocity"] = joints.velocity
    if joints.effort is not None:
        extras["joint_effort"] = joints.effort
    if observation.ee_pose is not None:
        pose = observation.ee_pose.pose
        extras["ee_pose"] = np.concatenate(
            [pose.position.as_array(), pose.orientation.as_array()]
        )
    force = getattr(robot, "gripper_contact_force", None)
    if force is not None:
        extras["gripper_force_n"] = force()
    metrics = getattr(policy, "metrics", None)
    if dataclasses.is_dataclass(metrics):
        for key, value in dataclasses.asdict(metrics).items():
            extras[f"policy.{key}"] = value
    for key, value in (info or {}).items():
        if isinstance(value, str | bool | int | float | np.number | np.bool_):
            extras[f"info.{key}"] = value
    return {key: _value(value) for key, value in extras.items()}
