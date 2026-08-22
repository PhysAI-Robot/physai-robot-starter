"""SO-101 robot factory kept behind the generic robot registry."""

from __future__ import annotations

from typing import Any

from ...sim import SO101PickPlaceEnv
from ...sim.env import EnvConfig


def make_so101(config: EnvConfig | None = None, **kwargs: Any) -> SO101PickPlaceEnv:
    """Build the current MuJoCo SO-101 pick-and-place environment."""
    if config is not None and kwargs:
        raise TypeError("pass either config or EnvConfig keyword fields, not both")
    return SO101PickPlaceEnv(config or EnvConfig(**kwargs))