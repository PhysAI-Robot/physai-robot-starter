"""Runtime registry for robot embodiment factories."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import RobotPort

RobotFactory = Callable[..., RobotPort]
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


def create_robot(name: str, *, adapter: str = "direct_mujoco", **kwargs: Any) -> RobotPort:
    """Create a robot through the selected simulation or hardware adapter."""
    _load_builtins()
    try:
        factory = _FACTORIES[name]
    except KeyError as exc:
        choices = ", ".join(available_robots())
        raise ValueError(f"unknown robot {name!r}; available: {choices}") from exc
    return factory(adapter=adapter, **kwargs)


def _load_builtins() -> None:
    if "so101" not in _FACTORIES:
        from .so101.factory import make_so101

        register_robot("so101", make_so101)
    if "turtlebot4" not in _FACTORIES:
        from .turtlebot.factory import make_turtlebot4

        register_robot("turtlebot4", make_turtlebot4)