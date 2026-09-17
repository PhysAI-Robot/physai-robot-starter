"""Task-specific MuJoCo scenes built from generic world primitives."""

from .common import (
    CommonSceneConfig,
    ManipulationSceneConfig,
    WorldSceneConfig,
    build_manipulation_spec,
)
from .pick_place_minimal import (
    PickPlaceMinimalSceneConfig,
)
from .pick_place_minimal import (
    build_spec as build_pick_place_spec,
)
from .registry import (
    SceneDefinition,
    available_scenes,
    create_scene,
    default_scene_for,
    get_scene_definition,
    register_scene,
)
from .sorting_minimal import SortingMinimalSceneConfig
from .sorting_minimal import build_spec as build_sorting_spec

__all__ = [
    "CommonSceneConfig",
    "ManipulationSceneConfig",
    "PickPlaceMinimalSceneConfig",
    "SceneDefinition",
    "SortingMinimalSceneConfig",
    "WorldSceneConfig",
    "available_scenes",
    "build_manipulation_spec",
    "build_pick_place_spec",
    "build_sorting_spec",
    "create_scene",
    "default_scene_for",
    "get_scene_definition",
    "register_scene",
]
