"""Task-specific MuJoCo scenes built from generic world primitives."""

from .common import (
    CommonSceneConfig,
    ManipulationSceneConfig,
    WorldSceneConfig,
    build_manipulation_spec,
)
from .pick_place_minimal import PickPlaceMinimalSceneConfig, build_spec as build_pick_place_spec
from .registry import (
    SceneDefinition,
    available_scenes,
    create_scene,
    get_scene_definition,
    register_scene,
)
from .sorting_minimal import SortingMinimalSceneConfig, build_spec as build_sorting_spec

__all__ = [
    "CommonSceneConfig",
    "WorldSceneConfig",
    "ManipulationSceneConfig",
    "build_manipulation_spec",
    "PickPlaceMinimalSceneConfig",
    "SortingMinimalSceneConfig",
    "build_pick_place_spec",
    "build_sorting_spec",
    "SceneDefinition",
    "available_scenes",
    "create_scene",
    "get_scene_definition",
    "register_scene",
]
