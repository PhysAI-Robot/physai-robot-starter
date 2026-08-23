"""Stable contracts between robot embodiments and the rest of the stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..contracts import Action, Observation


@dataclass(frozen=True)
class RobotSpec:
    """Describes an embodiment without exposing its simulator or hardware API."""

    name: str
    kind: str
    joint_names: tuple[str, ...] = ()
    action_modes: tuple[str, ...] = ("joint_position",)
    observation_modalities: tuple[str, ...] = ("state", "images")
    capabilities: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

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


class RobotEnv(Protocol):
    """Minimum environment surface a policy runner needs."""

    robot_spec: RobotSpec

    def reset(self, seed: int | None = None) -> Observation: ...

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict]: ...

    def close(self) -> None: ...