"""Registry for task-scene configurations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .common import ManipulationSceneConfig

SceneFactory = Callable[..., ManipulationSceneConfig]


@dataclass(frozen=True)
class SceneDefinition:
    """Configuration factory plus the compatibility it explicitly supports."""

    name: str
    factory: SceneFactory
    robot_kinds: tuple[str, ...]
    task_names: tuple[str, ...]

    def supports(self, robot_kind: str, task_name: str | None = None) -> bool:
        return robot_kind in self.robot_kinds and (
            task_name is None or task_name in self.task_names
        )


_DEFINITIONS: dict[str, SceneDefinition] = {}
_BUILTINS_LOADED = False


def register_scene(name: str, definition: SceneDefinition) -> SceneDefinition:
    if name in _DEFINITIONS:
        raise ValueError(f"scene {name!r} is already registered")
    if definition.name != name:
        raise ValueError(
            f"scene definition name {definition.name!r} does not match {name!r}"
        )
    _DEFINITIONS[name] = definition
    return definition


def available_scenes() -> tuple[str, ...]:
    _load_builtins()
    return tuple(sorted(_DEFINITIONS))


def get_scene_definition(name: str) -> SceneDefinition:
    _load_builtins()
    try:
        return _DEFINITIONS[name]
    except KeyError as exc:
        choices = ", ".join(available_scenes())
        raise ValueError(f"unknown scene {name!r}; available: {choices}") from exc


def create_scene(name: str, **kwargs: Any) -> ManipulationSceneConfig:
    """Create a scene config without importing a concrete scene at call sites."""
    return get_scene_definition(name).factory(**kwargs)


def _load_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return

    from .pick_place_minimal import PickPlaceMinimalSceneConfig
    from .sorting_minimal import SortingMinimalSceneConfig

    register_scene(
        "pick_place_minimal",
        SceneDefinition(
            name="pick_place_minimal",
            factory=PickPlaceMinimalSceneConfig,
            robot_kinds=("fixed_base_manipulator",),
            task_names=("pick_place",),
        ),
    )
    register_scene(
        "sorting_minimal",
        SceneDefinition(
            name="sorting_minimal",
            factory=SortingMinimalSceneConfig,
            robot_kinds=("fixed_base_manipulator",),
            task_names=("sorting",),
        ),
    )
    _BUILTINS_LOADED = True
