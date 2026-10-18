"""ViperX 300 S robot factory kept behind the generic robot registry."""

from __future__ import annotations

from typing import Any

from ..adapters import create_adapter
from ..base import RobotPort
from .mujoco_env import EnvConfig, ViperX300SEnv


def viperx300s_env_config(*, simulator: str = "mujoco", **kwargs: Any) -> Any:
    """The ViperX 300 S environment config type for one simulator engine."""
    if simulator == "isaac":
        from .isaac_env import ViperX300SIsaacEnvConfig

        return ViperX300SIsaacEnvConfig(**kwargs)
    if simulator == "mujoco":
        return EnvConfig(**kwargs)
    raise ValueError(f"unknown simulator {simulator!r}; available: isaac, mujoco")


def make_viperx300s(
    config: Any = None,
    *,
    adapter: str = "direct",
    simulator: str = "mujoco",
    transport: Any = None,
    hardware: RobotPort | None = None,
    codec: Any = None,
    **kwargs: Any,
) -> RobotPort:
    """Build the ALOHA follower arm through the selected robot port adapter."""
    if config is not None and kwargs:
        raise TypeError("pass either config or a config's keyword fields, not both")
    if adapter == "ros2_real":
        raise ValueError("ViperX 300 S has no ros2_real adapter yet")
    if simulator == "isaac":
        from .isaac_env import ViperX300SIsaacEnv

        direct: RobotPort = ViperX300SIsaacEnv(
            config or viperx300s_env_config(simulator="isaac", **kwargs)
        )
    elif simulator == "mujoco":
        direct = ViperX300SEnv(
            config or viperx300s_env_config(simulator="mujoco", **kwargs)
        )
    else:
        raise ValueError(f"unknown simulator {simulator!r}; available: isaac, mujoco")
    return create_adapter(
        adapter, direct, transport=transport, hardware=hardware, codec=codec
    )
