"""SO-101 robot factory kept behind the generic robot registry."""

from __future__ import annotations

from typing import Any

from ..adapters import create_adapter
from ..base import RobotPort
from .mujoco_env import EnvConfig, SO101Env


def so101_env_config(*, simulator: str = "mujoco", **kwargs: Any) -> Any:
    """The SO-101 environment config type for one simulator engine."""
    if simulator == "isaac":
        from .isaac_env import IsaacEnvConfig

        return IsaacEnvConfig(**kwargs)
    if simulator == "mujoco":
        return EnvConfig(**kwargs)
    raise ValueError(f"unknown simulator {simulator!r}; available: isaac, mujoco")


def make_so101(
    config: Any = None,
    *,
    adapter: str = "direct",
    simulator: str = "mujoco",
    transport: Any = None,
    hardware: RobotPort | None = None,
    codec: Any = None,
    **kwargs: Any,
) -> RobotPort:
    """Build the SO-101 through the selected simulator and robot port adapter.

    `simulator` picks which physics engine builds the direct port
    (`mujoco` -> `SO101Env`, `isaac` -> `SO101IsaacEnv`); `adapter` then
    wraps whichever one was built. `DirectAdapter` only wraps a generic
    `RobotPort` (observe/reset/step/send_action/close plus a safety gate)
    and has no simulator-specific behavior itself, so it is the right
    wrapper for either simulator — there is no separate "direct_isaac"
    adapter to register.
    """
    if config is not None and kwargs:
        raise TypeError("pass either config or a config's keyword fields, not both")
    if adapter == "ros2_real":
        if config is not None or kwargs:
            raise TypeError(
                "robot config is not used by adapter='ros2_real'; "
                "configure the hardware port instead"
            )
        return create_adapter(
            adapter, None, transport=transport, hardware=hardware, codec=codec
        )
    if simulator == "isaac":
        from .isaac_env import SO101IsaacEnv

        direct: RobotPort = SO101IsaacEnv(
            config or so101_env_config(simulator="isaac", **kwargs)
        )
    elif simulator == "mujoco":
        direct = SO101Env(config or so101_env_config(simulator="mujoco", **kwargs))
    else:
        raise ValueError(f"unknown simulator {simulator!r}; available: isaac, mujoco")
    return create_adapter(
        adapter, direct, transport=transport, hardware=hardware, codec=codec
    )
