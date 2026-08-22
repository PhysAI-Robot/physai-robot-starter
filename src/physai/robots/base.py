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
    metadata: dict[str, Any] = field(default_factory=dict)


class RobotEnv(Protocol):
    """Minimum environment surface a policy runner needs."""

    robot_spec: RobotSpec

    def reset(self, seed: int | None = None) -> Observation: ...

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict]: ...

    def close(self) -> None: ...