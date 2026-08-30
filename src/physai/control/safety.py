"""Safety validation between a policy and a robot port."""

from __future__ import annotations

import time

import numpy as np

from ..contracts import Action, Observation
from ..robots.base import RobotSpec


class SafetyController:
    """Validate commands before they reach simulation or hardware.

    The controller is deliberately independent from MuJoCo and ROS2. A
    deployment can therefore use the same checks in direct simulation, a ROS2
    bridge, and a hardware driver.
    """

    def __init__(
        self,
        robot_spec: RobotSpec,
        *,
        max_action_age: float | None = 0.5,
        future_tolerance: float = 0.1,
    ) -> None:
        if max_action_age is not None and max_action_age <= 0:
            raise ValueError("max_action_age must be positive or None")
        if future_tolerance < 0:
            raise ValueError("future_tolerance must be non-negative")
        self.robot_spec = robot_spec
        self.max_action_age = max_action_age
        self.future_tolerance = future_tolerance

    def validate(
        self,
        observation: Observation | None,
        action: Action,
        *,
        now: float | None = None,
    ) -> Action:
        """Return a validated action or raise before it reaches the robot."""
        self.robot_spec.validate_action(action)
        current_time = time.time() if now is None else float(now)
        if action.stamp is not None:
            age = current_time - action.stamp
            if age < -self.future_tolerance:
                raise ValueError("action timestamp is too far in the future")
            if self.max_action_age is not None and age > self.max_action_age:
                raise ValueError("action is stale")

        if action.mode != "joint_position":
            return action

        names = action.joint_names or self.robot_spec.action_joint_names
        values = action.joint_position
        if values is None:
            raise ValueError("joint-position action has no values")
        if names and len(names) != values.size:
            raise ValueError("joint names and action values have different lengths")

        for index, name in enumerate(names):
            limit = self.robot_spec.joint_limits.get(name)
            if limit is not None and not limit[0] <= values[index] <= limit[1]:
                raise ValueError(
                    f"action for joint {name!r} is outside limits {limit}: "
                    f"{values[index]}"
                )
            if observation is not None:
                max_delta = self.robot_spec.max_joint_delta.get(name)
                if max_delta is not None:
                    current = observation.joint_state.get(name)
                    if abs(values[index] - current) > max_delta:
                        raise ValueError(
                            f"action for joint {name!r} exceeds max step {max_delta}"
                        )
        return action
