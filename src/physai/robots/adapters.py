"""Robot adapters for synchronous simulation and future transport backends."""

from __future__ import annotations

from typing import Any

from ..contracts import Action, Observation
from .base import RobotPort, RobotSpec

ADAPTER_NAMES = ("direct_mujoco", "ros2_mujoco", "ros2_hardware")


class DirectMuJoCoAdapter:
    """Expose a MuJoCo-backed robot through the generic robot port."""

    def __init__(self, environment: RobotPort) -> None:
        self._environment = environment

    @property
    def robot_spec(self) -> RobotSpec:
        return self._environment.robot_spec

    def reset(self, seed: int | None = None) -> Observation:
        return self._environment.reset(seed=seed)

    def observe(self) -> Observation:
        return self._environment.observe()

    def send_action(self, action: Action) -> None:
        self.robot_spec.validate_action(action)
        self._environment.send_action(action)

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict]:
        self.robot_spec.validate_action(action)
        return self._environment.step(action)

    def close(self) -> None:
        self._environment.close()

    def __getattr__(self, name: str) -> Any:
        """Keep the Phase 0 convenience surface available during migration."""
        return getattr(self._environment, name)


def select_adapter(
    name: str,
    direct: RobotPort,
    *,
    transport: Any = None,
    hardware: RobotPort | None = None,
) -> RobotPort:
    """Select a robot adapter without changing policy or task code."""
    if name == "direct_mujoco":
        return DirectMuJoCoAdapter(direct)
    if name == "ros2_mujoco":
        if transport is None:
            raise ValueError("adapter='ros2_mujoco' requires a ROS2 transport")
        from ..bridge.adapters import ROS2MuJoCoAdapter

        return ROS2MuJoCoAdapter(direct, transport)
    if name == "ros2_hardware":
        if transport is None:
            raise ValueError("adapter='ros2_hardware' requires a ROS2 transport")
        if hardware is None:
            raise ValueError("adapter='ros2_hardware' requires a hardware port")
        from ..bridge.adapters import ROS2HardwareAdapter

        return ROS2HardwareAdapter(hardware, transport)
    choices = ", ".join(ADAPTER_NAMES)
    raise ValueError(f"unknown adapter {name!r}; available: {choices}")
