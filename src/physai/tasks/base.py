"""Task contract shared by simulation and hardware backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol


class TaskBackend(Protocol):
    """State provider consumed by task evaluation and reward logic."""


class Task(ABC):
    """Own task state, reward, and termination; never choose robot actions."""

    name = "task"
    required_capabilities: tuple[str, ...] = ()
    required_action_modes: tuple[str, ...] = ()

    @abstractmethod
    def evaluate(self, backend: TaskBackend) -> dict:
        """Return task metrics from the backend's current state."""

    @abstractmethod
    def reward(self, backend: TaskBackend, info: dict | None = None) -> float:
        """Return the scalar reward for the current task state."""

    def reset(self, backend: TaskBackend, rng: object) -> None:
        """Reset task-owned state. Backends may perform physical reset work."""

    def terminated(self, backend: TaskBackend, info: dict) -> bool:
        return bool(info.get("success") or info.get("failure"))
