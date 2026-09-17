"""SO-101-owned policy factories for the shared policy registry."""

from __future__ import annotations

from typing import Any

from .expert import SO101PickPlaceExpert
from .visual_servo import SO101VisualServoPolicy


def make_scripted_policy(*, env, cfg: Any = None, **_: Any) -> SO101PickPlaceExpert:
    """Build the deterministic SO-101 pick-and-place expert."""
    return SO101PickPlaceExpert(env.kin, env, cfg=cfg)


def make_visual_servo_policy(
    *, env, cfg: Any = None, **kwargs: Any
) -> SO101VisualServoPolicy:
    """Build the deterministic SO-101 camera-feedback baseline."""
    options = dict(kwargs)
    if cfg is not None:
        options.update(cfg if isinstance(cfg, dict) else vars(cfg))
    return SO101VisualServoPolicy(env, **options)
