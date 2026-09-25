"""Registry for planner implementations.

Mirrors `tasks/registry.py`. `ScriptedPlanner` is a core baseline and
registers here directly; a research planner (e.g.
`research/vlm_planners/sorting_planner.py`'s `SortingPlanner`) registers
itself under its own name when its module is imported, so this registry
never imports research code.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import Planner

PlannerFactory = Callable[..., Planner]
_FACTORIES: dict[str, PlannerFactory] = {}
_BUILTINS_LOADED = False


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
    # A flag, not "is anything registered": a research planner that registers
    # itself on import may run before the first lookup.
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    from .base import ScriptedPlanner

    register_planner("scripted_planner", ScriptedPlanner)
    _BUILTINS_LOADED = True
