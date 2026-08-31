"""SO-101 robot factory kept behind the generic robot registry."""

from __future__ import annotations

from typing import Any

from ..adapters import select_adapter
from ..base import RobotPort
from .env import EnvConfig, SO101Env


def make_so101(
    config: EnvConfig | None = None,
    *,
    adapter: str = "direct_mujoco",
    transport: Any = None,
    hardware: RobotPort | None = None,
    codec: Any = None,
    **kwargs: Any,
) -> RobotPort:
    """Build the SO-101 through the selected robot port adapter."""
    if config is not None and kwargs:
        raise TypeError("pass either config or EnvConfig keyword fields, not both")
    if adapter == "ros2_hardware":
        if config is not None or kwargs:
            raise TypeError(
                "robot config is not used by adapter='ros2_hardware'; "
                "configure the hardware port instead"
            )
        return select_adapter(
            adapter, None, transport=transport, hardware=hardware, codec=codec
        )
    direct = SO101Env(config or EnvConfig(**kwargs))
    return select_adapter(
        adapter, direct, transport=transport, hardware=hardware, codec=codec
    )