"""Registry for task definitions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import Task

TaskFactory = Callable[..., Task]
_FACTORIES: dict[str, TaskFactory] = {}


def register_task(name: str, factory: TaskFactory) -> TaskFactory:
    if name in _FACTORIES:
        raise ValueError(f"task {name!r} is already registered")
    _FACTORIES[name] = factory
    return factory


def available_tasks() -> tuple[str, ...]:
    _load_builtins()
    return tuple(sorted(_FACTORIES))


def create_task(name: str, **kwargs: Any) -> Task:
    _load_builtins()
    try:
        return _FACTORIES[name](**kwargs)
    except KeyError as exc:
        choices = ", ".join(available_tasks())
        raise ValueError(f"unknown task {name!r}; available: {choices}") from exc


def _load_builtins() -> None:
    if _FACTORIES:
        return
    from .pick_place import PickPlaceTask
    from .sorting import SortingTask

    register_task("pick_place", PickPlaceTask)
    register_task("sorting", SortingTask)
