"""SO-101 robot factory kept behind the generic robot registry."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..adapters import create_adapter
from ..base import RobotPort
from ..rendering import attach_renderer, start_isaac_first
from .mujoco_env import EnvConfig, SO101Env


def so101_env_config(*, simulator: str = "mujoco", **kwargs: Any) -> Any:
    """The SO-101 environment config type for one simulator engine."""
    if simulator == "isaac":
        from .isaac_env import IsaacEnvConfig

        return IsaacEnvConfig(**kwargs)
    if simulator in ("mujoco", "newton"):
        return EnvConfig(**kwargs)
    raise ValueError(
        f"unknown simulator {simulator!r}; available: isaac, mujoco, newton"
    )


def _render_only_isaac_env(scene: Any) -> Any:
    from .isaac_env import IsaacEnvConfig, SO101IsaacEnv

    return SO101IsaacEnv(
        IsaacEnvConfig(
            scene=scene,
            camera_resolution=scene.camera_resolution,
            cameras=(),
            render=True,
        )
    )


def make_so101(
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
    """Build the SO-101 through the selected simulator and robot port adapter.

    `simulator` picks which physics engine builds the direct port and
    `renderer` which engine draws its cameras (default: the simulator's own),
    so any pairing works. The adapter wrapper (`ros2_sim`, `ros2_real`,
    `direct`) takes the direct port and has no simulator-specific behavior.
    """
    if config is not None and kwargs:
        raise TypeError("pass either config or a config's keyword fields, not both")
    config = config or so101_env_config(simulator=simulator, **kwargs)
    start_isaac_first(simulator, renderer)
    foreign = renderer not in (None, simulator) and config.render
    if foreign and simulator == "isaac":
        config = replace(config, render=False)  # another renderer draws the cameras
    if simulator == "isaac":
        from .isaac_env import SO101IsaacEnv

        direct: RobotPort = SO101IsaacEnv(config)
    elif simulator == "mujoco":
        direct = SO101Env(config)
    elif simulator == "newton":
        from .newton_env import SO101NewtonEnv

        direct = SO101NewtonEnv(config)
    else:
        raise ValueError(
            f"unknown simulator {simulator!r}; available: isaac, mujoco, newton"
        )
    if foreign:
        attach_renderer(direct, simulator, renderer, _render_only_isaac_env)
    return create_adapter(
        adapter, direct, transport=transport, hardware=hardware, codec=codec
    )
