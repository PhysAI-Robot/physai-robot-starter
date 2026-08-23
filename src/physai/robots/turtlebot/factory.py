"""TurtleBot4 factory kept behind the generic robot registry."""

from __future__ import annotations

from typing import Any

from .env import TurtleBot4Config, TurtleBot4Env


def make_turtlebot4(config: TurtleBot4Config | None = None, **kwargs: Any) -> TurtleBot4Env:
    """Build a TurtleBot4 differential-drive environment."""
    if config is not None and kwargs:
        raise TypeError("pass either config or TurtleBot4Config keyword fields, not both")
    return TurtleBot4Env(config or TurtleBot4Config(**kwargs))