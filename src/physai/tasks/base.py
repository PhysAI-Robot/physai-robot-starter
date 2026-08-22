"""Task contract shared by simulation and hardware backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Task(ABC):
    """Own task state, reward, and termination; never choose robot actions."""

    name = "task"

    @abstractmethod
    def evaluate(self, backend: Any) -> dict:
        """Return task metrics from the backend's current state."""

    @abstractmethod
    def reward(self, backend: Any, info: dict | None = None) -> float:
        """Return the scalar reward for the current task state."""

    def reset(self, backend: Any, rng: Any) -> None:
        """Reset task-owned state. Backends may perform physical reset work."""

    def terminated(self, backend: Any, info: dict) -> bool:
        return bool(info.get("success") or info.get("failure"))
