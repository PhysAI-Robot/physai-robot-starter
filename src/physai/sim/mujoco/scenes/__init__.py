"""Task-specific MuJoCo scenes built from generic world primitives."""

from .common import (
    ManipulationSceneConfig,
    WorldSceneConfig,
    build_manipulation_spec,
    export_xml,
)
from .pick_place_minimal import PickPlaceMinimalSceneConfig
from .registry import (
    SceneDefinition,
    available_scenes,
    create_scene,
    default_scene_for,
    get_scene_definition,
    register_scene,
)
from .sorting_minimal import SortingMinimalSceneConfig

__all__ = [
    "ManipulationSceneConfig",
    "PickPlaceMinimalSceneConfig",
    "SceneDefinition",
    "SortingMinimalSceneConfig",
    "WorldSceneConfig",
    "available_scenes",
    "build_manipulation_spec",
    "create_scene",
    "default_scene_for",
    "export_xml",
    "get_scene_definition",
    "register_scene",
]
