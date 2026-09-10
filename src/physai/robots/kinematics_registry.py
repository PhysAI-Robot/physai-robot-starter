"""Registry for embodiment-specific kinematics adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import KinematicsPort

KinematicsFactory = Callable[..., KinematicsPort]
_FACTORIES: dict[str, KinematicsFactory] = {}


def register_kinematics_adapter(name: str, factory: KinematicsFactory) -> None:
    """Register one robot-owned kinematics implementation."""
    if not name or not name.strip():
        raise ValueError("kinematics adapter name must not be empty")
    if name in _FACTORIES:
        raise ValueError(f"kinematics adapter {name!r} is already registered")
    _FACTORIES[name] = factory


def create_kinematics_adapter(name: str, *args: Any, **kwargs: Any) -> KinematicsPort:
    """Construct a registered adapter without assuming a robot embodiment."""
    try:
        factory = _FACTORIES[name]
    except KeyError as exc:
        available = ", ".join(sorted(_FACTORIES)) or "none"
        raise ValueError(
            f"unknown kinematics adapter {name!r}; available: {available}"
        ) from exc
    return factory(*args, **kwargs)


def available_kinematics_adapters() -> tuple[str, ...]:
    return tuple(sorted(_FACTORIES))


__all__ = [
    "KinematicsFactory",
    "available_kinematics_adapters",
    "create_kinematics_adapter",
    "register_kinematics_adapter",
]