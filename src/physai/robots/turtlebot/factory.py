"""TurtleBot4 factory kept behind the generic robot registry."""

from __future__ import annotations

from typing import Any

from ..adapters import select_adapter
from ..base import RobotPort
from .env import TurtleBot4Config, TurtleBot4Env


def make_turtlebot4(
    config: TurtleBot4Config | None = None,
    *,
    adapter: str = "direct_mujoco",
    transport: Any = None,
    hardware: RobotPort | None = None,
    codec: Any = None,
    **kwargs: Any,
) -> RobotPort:
    """Build TurtleBot4 through the selected robot port adapter."""
    if config is not None and kwargs:
        raise TypeError("pass either config or TurtleBot4Config keyword fields, not both")
    if adapter == "ros2_hardware":
        if config is not None or kwargs:
            raise TypeError(
                "robot config is not used by adapter='ros2_hardware'; "
                "configure the hardware port instead"
            )
        return select_adapter(
            adapter, None, transport=transport, hardware=hardware, codec=codec
        )
    direct = TurtleBot4Env(config or TurtleBot4Config(**kwargs))
    return select_adapter(
        adapter, direct, transport=transport, hardware=hardware, codec=codec
    )