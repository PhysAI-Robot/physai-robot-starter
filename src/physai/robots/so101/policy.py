"""SO-101-owned policy factories for the shared policy registry."""

from __future__ import annotations

from typing import Any

from .expert import SO101PickPlaceExpert


def make_scripted_policy(*, env, cfg: Any = None, **_: Any) -> SO101PickPlaceExpert:
    """Build the deterministic SO-101 pick-and-place expert."""
    return SO101PickPlaceExpert(env.kin, env, cfg=cfg)
