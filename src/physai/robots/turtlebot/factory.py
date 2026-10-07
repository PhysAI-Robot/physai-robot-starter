"""TurtleBot4 factory kept behind the generic robot registry."""

from __future__ import annotations

from typing import Any

from ..adapters import create_adapter
from ..base import RobotPort
from .env import TurtleBot4Config, TurtleBot4Env


def make_turtlebot4(
    config: TurtleBot4Config | None = None,
    *,
    adapter: str = "direct",
    simulator: str = "mujoco",
    transport: Any = None,
    hardware: RobotPort | None = None,
    codec: Any = None,
    **kwargs: Any,
) -> RobotPort:
    """Build TurtleBot4 through the selected robot port adapter.

    `simulator` is accepted (not just ignored) because the registry always
    passes it; `RobotDescriptor.simulators` defaults to `("mujoco",)` for
    TurtleBot4, so it is already validated to be `"mujoco"` by the time it
    gets here — there is no second engine for it to pick between yet.
    """
    if adapter == "ros2_real":
        raise ValueError(
            "adapter='ros2_real' is not supported for turtlebot4; "
            "the hardware-specific mobile-base adapter is not implemented"
        )
    if config is not None and kwargs:
        raise TypeError(
            "pass either config or TurtleBot4Config keyword fields, not both"
        )
    direct = TurtleBot4Env(config or TurtleBot4Config(**kwargs))
    return create_adapter(
        adapter, direct, transport=transport, hardware=hardware, codec=codec
    )
