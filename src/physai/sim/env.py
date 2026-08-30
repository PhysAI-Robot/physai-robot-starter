"""Compatibility facade for the SO-101 environment.

The concrete embodiment implementation lives in ``physai.robots.so101``.
This module remains importable for Phase 0 callers that used the original
``physai.sim.env`` path.
"""

from ..robots.so101.env import EnvConfig, HOME_QPOS, SO101PickPlaceEnv

__all__ = ["EnvConfig", "HOME_QPOS", "SO101PickPlaceEnv"]
