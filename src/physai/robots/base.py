"""Stable contracts between robot embodiments and the rest of the stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from ..contracts import Action, Observation, PoseStamped


@dataclass(frozen=True)
class RobotSpec:
    """Describes an embodiment without exposing its simulator or hardware API."""

    name: str
    kind: str
    joint_names: tuple[str, ...] = ()
    action_joint_names: tuple[str, ...] = ()
    action_modes: tuple[str, ...] = ("joint_position",)
    observation_modalities: tuple[str, ...] = ("state", "images")
    capabilities: tuple[str, ...] = ()
    joint_limits: dict[str, tuple[float, float]] = field(default_factory=dict)
    max_joint_delta: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    joint_state_frame: str | None = None
    action_frame: str | None = None
    camera_frames: dict[str, str] = field(default_factory=dict)

    def supports(self, *capabilities: str) -> bool:
        """Return whether this embodiment provides every requested capability."""
        available = set(self.capabilities)
        return all(capability in available for capability in capabilities)

    def require(self, *capabilities: str) -> None:
        """Raise a clear error when a workflow needs unsupported capabilities."""
        missing = [capability for capability in capabilities
                   if capability not in self.capabilities]
        if missing:
            requested = ", ".join(missing)
            raise ValueError(f"robot {self.name!r} does not support: {requested}")

    def validate_action(self, action: Action) -> None:
        """Reject commands that do not match this embodiment's capabilities."""
        mode = action.mode
        if mode is None:
            raise ValueError(f"robot {self.name!r} requires an action mode")
        if mode not in self.action_modes:
            choices = ", ".join(self.action_modes)
            raise ValueError(
                f"robot {self.name!r} does not support action mode {mode!r}; "
                f"available: {choices}"
            )
        if mode == "joint_position" and self.action_joint_names:
            size = action.joint_position.size
            expected = len(self.action_joint_names)
            if size != expected:
                raise ValueError(
                    f"robot {self.name!r} expects {expected} joint targets, got {size}"
                )
            if action.joint_names is not None and action.joint_names != self.action_joint_names:
                raise ValueError(
                    f"robot {self.name!r} expects joint order {self.action_joint_names}, "
                    f"got {action.joint_names}"
                )
        if mode == "twist":
            if action.ee_twist is None or not action.ee_twist.frame_id:
                raise ValueError("twist action frame_id must not be empty")
            if self.action_frame is not None and action.ee_twist.frame_id != self.action_frame:
                raise ValueError(
                    f"robot {self.name!r} expects twist frame {self.action_frame!r}, "
                    f"got {action.ee_twist.frame_id!r}"
                )
        values = action.joint_position if mode == "joint_position" else action.ee_twist.as_array()
        if not np.isfinite(values).all():
            raise ValueError("action contains non-finite values")

        if action.stamp is not None and not np.isfinite(action.stamp):
            raise ValueError("action stamp must be finite")

    def validate_observation(self, observation: Observation) -> None:
        """Validate observation names, units, timestamps, and frame IDs."""
        observation.validate(
            expected_joint_names=self.joint_names,
            expected_joint_frame=self.joint_state_frame,
            expected_camera_frames=self.camera_frames,
        )

    def validate_task(self, task: object) -> None:
        """Validate a task's declared capabilities before an episode starts."""
        required = getattr(task, "required_capabilities", ())
        self.require(*required)
        modes = tuple(getattr(task, "required_action_modes", ()))
        missing_modes = [mode for mode in modes if mode not in self.action_modes]
        if missing_modes:
            requested = ", ".join(missing_modes)
            raise ValueError(
                f"robot {self.name!r} does not support task action modes: {requested}"
            )


class RobotPort(Protocol):
    """Application boundary shared by simulation and hardware adapters."""

    robot_spec: RobotSpec

    def reset(self, seed: int | None = None) -> Observation: ...

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict]: ...

    def observe(self) -> Observation: ...

    def send_action(self, action: Action) -> None: ...

    def close(self) -> None: ...


class RobotEnv(RobotPort, Protocol):
    """Backward-compatible synchronous environment contract."""


class KinematicsPort(Protocol):
    """Robot kinematics operations required by Cartesian control code."""

    def fk(self, state: Any) -> PoseStamped: ...

    def ik(self, target_pos: Any, approach_dir: Any = None,
           q_init: Any = None, **kwargs: Any) -> Any: ...

    def ik_pinch(self, object_center: Any, approach_dir: Any = None,
                 q_init: Any = None, **kwargs: Any) -> Any: ...

    def site_jacobian(self, state: Any) -> np.ndarray: ...

    def pinch_center(self, state: Any, offset: Any = None) -> np.ndarray: ...

    def clip_to_limits(self, joint_positions: Any) -> np.ndarray: ...