"""Planner backend registry.

New VLM approaches register a factory here; callers do not need to know the
backend's SDK or constructor details.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import Planner

PlannerFactory = Callable[..., Planner]
_FACTORIES: dict[str, PlannerFactory] = {}


def register_planner(name: str, factory: PlannerFactory) -> PlannerFactory:
    if name in _FACTORIES:
        raise ValueError(f"planner {name!r} is already registered")
    _FACTORIES[name] = factory
    return factory


def available_planners() -> tuple[str, ...]:
    _load_builtins()
    return tuple(sorted(_FACTORIES))


def create_planner(name: str, **kwargs: Any) -> Planner:
    _load_builtins()
    try:
        return _FACTORIES[name](**kwargs)
    except KeyError as exc:
        choices = ", ".join(available_planners())
        raise ValueError(f"unknown planner {name!r}; available: {choices}") from exc


def _load_builtins() -> None:
    if _FACTORIES:
        return
    from .base import ScriptedPlanner
    from .claude_vlm import ClaudePlanner
    from .smolvlm import SmolVLMPlanner

    register_planner("claude", ClaudePlanner)
    register_planner("smolvlm", SmolVLMPlanner)
    register_planner("scripted", ScriptedPlanner)