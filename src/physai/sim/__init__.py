from .core import MuJoCoSimulationCore
from ..robots.so101.kinematics import ArmKinematics, IKResult, top_down_quat
from .scene import SceneConfig, build_model, build_spec, export_xml
from .scenes import (
    PickPlaceMinimalSceneConfig,
    CommonSceneConfig,
    ManipulationSceneConfig,
    SortingMinimalSceneConfig,
    WorldSceneConfig,
    available_scenes,
    build_manipulation_spec,
    create_scene,
)

__all__ = [
    "ArmKinematics",
    "EnvConfig",
    "HOME_QPOS",
    "IKResult",
    "MuJoCoSimulationCore",
    "CommonSceneConfig",
    "WorldSceneConfig",
    "ManipulationSceneConfig",
    "build_manipulation_spec",
    "PickPlaceMinimalSceneConfig",
    "SortingMinimalSceneConfig",
    "available_scenes",
    "create_scene",
    "SO101PickPlaceEnv",
    "SceneConfig",
    "build_model",
    "build_spec",
    "export_xml",
    "top_down_quat",
]


def __getattr__(name: str):
    """Load legacy SO-101 exports only when a caller requests them."""
    if name in {"EnvConfig", "HOME_QPOS", "SO101PickPlaceEnv"}:
        from .env import EnvConfig, HOME_QPOS, SO101PickPlaceEnv

        return {
            "EnvConfig": EnvConfig,
            "HOME_QPOS": HOME_QPOS,
            "SO101PickPlaceEnv": SO101PickPlaceEnv,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
