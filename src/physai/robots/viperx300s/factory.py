"""ViperX 300 S robot factory kept behind the generic robot registry."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..adapters import create_adapter
from ..base import RobotPort
from ..rendering import attach_renderer, start_isaac_first
from .mujoco_env import EnvConfig, ViperX300SEnv


def viperx300s_env_config(*, simulator: str = "mujoco", **kwargs: Any) -> Any:
    """The ViperX 300 S environment config type for one simulator engine."""
    if simulator == "isaac":
        from .isaac_env import ViperX300SIsaacEnvConfig

        return ViperX300SIsaacEnvConfig(**kwargs)
    if simulator in ("mujoco", "newton"):
        return EnvConfig(**kwargs)
    raise ValueError(
        f"unknown simulator {simulator!r}; available: isaac, mujoco, newton"
    )


def _render_only_isaac_env(scene: Any) -> Any:
    from .isaac_env import ViperX300SIsaacEnv, ViperX300SIsaacEnvConfig

    return ViperX300SIsaacEnv(
        ViperX300SIsaacEnvConfig(
            scene=scene,
            camera_resolution=scene.camera_resolution,
            cameras=(),
            render=True,
        )
    )


def make_viperx300s(
    config: Any = None,
    *,
    adapter: str = "direct",
    simulator: str = "mujoco",
    renderer: str | None = None,
    transport: Any = None,
    hardware: RobotPort | None = None,
    codec: Any = None,
    **kwargs: Any,
) -> RobotPort:
    """Build the ALOHA follower arm through the selected robot port adapter.

    `simulator` picks the physics engine and `renderer` the engine that draws
    its cameras (default: the simulator's own).
    """
    if config is not None and kwargs:
        raise TypeError("pass either config or a config's keyword fields, not both")
    if adapter == "ros2_real":
        raise ValueError("ViperX 300 S has no ros2_real adapter yet")
    config = config or viperx300s_env_config(simulator=simulator, **kwargs)
    start_isaac_first(simulator, renderer)
    foreign = renderer not in (None, simulator) and config.render
    if foreign and simulator == "isaac":
        config = replace(config, render=False)  # another renderer draws the cameras
    if simulator == "isaac":
        from .isaac_env import ViperX300SIsaacEnv

        direct: RobotPort = ViperX300SIsaacEnv(config)
    elif simulator == "mujoco":
        direct = ViperX300SEnv(config)
    elif simulator == "newton":
        from .newton_env import ViperX300SNewtonEnv

        direct = ViperX300SNewtonEnv(config)
    else:
        raise ValueError(
            f"unknown simulator {simulator!r}; available: isaac, mujoco, newton"
        )
    if foreign:
        attach_renderer(direct, simulator, renderer, _render_only_isaac_env)
    return create_adapter(
        adapter, direct, transport=transport, hardware=hardware, codec=codec
    )
