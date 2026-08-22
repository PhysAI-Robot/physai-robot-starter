"""Runtime registry for robot embodiment factories."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import RobotEnv

RobotFactory = Callable[..., RobotEnv]
_FACTORIES: dict[str, RobotFactory] = {}


def register_robot(name: str, factory: RobotFactory) -> RobotFactory:
    """Register a robot factory under a stable configuration name."""
    if name in _FACTORIES:
        raise ValueError(f"robot {name!r} is already registered")
    _FACTORIES[name] = factory
    return factory


def available_robots() -> tuple[str, ...]:
    _load_builtins()
    return tuple(sorted(_FACTORIES))


def create_robot(name: str, **kwargs: Any) -> RobotEnv:
    _load_builtins()
    try:
        factory = _FACTORIES[name]
    except KeyError as exc:
        choices = ", ".join(available_robots())
        raise ValueError(f"unknown robot {name!r}; available: {choices}") from exc
    return factory(**kwargs)


def _load_builtins() -> None:
    if "so101" not in _FACTORIES:
        from .so101.factory import make_so101

        register_robot("so101", make_so101)