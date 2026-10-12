"""Runtime registry for control policy factories.

The registry is the policy composition boundary. CLI code selects a stable
name and supplies runtime dependencies; it does not import concrete policy
implementations or know their constructor details.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import Policy

PolicyFactory = Callable[..., Policy]
_FACTORIES: dict[str, PolicyFactory] = {}
_BUILTINS_LOADED = False


def register_policy(name: str, factory: PolicyFactory) -> PolicyFactory:
    """Register a policy factory under a stable configuration name."""
    if name in _FACTORIES:
        raise ValueError(f"policy {name!r} is already registered")
    _FACTORIES[name] = factory
    return factory


def available_policies(robot: str | None = None) -> tuple[str, ...]:
    """Policy names available to the current runtime: the generic ones plus the
    robot-owned ones (of `robot`, or of any robot when it is None)."""
    from ..robots.registry import robot_policies

    _load_builtins()
    return tuple(sorted({*_FACTORIES, *robot_policies(robot)}))


def create_policy(name: str, **kwargs: Any) -> Policy:
    """Create a policy without exposing its concrete implementation.

    A name that is not a generic policy falls back to the policy the `env`
    robot owns under that name (so101's "demo", or one a plugin registered).
    """
    from ..robots.registry import create_robot_policy, has_robot_policy

    _load_builtins()
    factory = _FACTORIES.get(name)
    if factory is not None:
        return factory(**kwargs)
    env = kwargs.get("env")
    robot = None if env is None else env.robot_spec.name
    if robot is not None and has_robot_policy(robot, name):
        return create_robot_policy(robot, name, **kwargs)
    choices = ", ".join(available_policies(robot))
    raise ValueError(
        f"unknown policy {name!r}; available: {choices}. A policy from your own "
        "module needs that module in the manifest's `plugins:` list"
    )


def _load_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return

    register_policy("constant", _make_constant)
    register_policy("constant_twist", _make_constant_twist)
    register_policy("replay", _make_replay)
    # Any other policy comes from a plugin module the manifest names; it
    # registers itself on import (see physai.plugins).
    _BUILTINS_LOADED = True


def _make_constant(*, env: Any = None, **_: Any) -> Policy:
    """The hold policy shaped for the robot's action space: a base holds a zero twist."""
    from .base import ConstantPolicy, ConstantTwistPolicy

    if env is not None and env.robot_spec.supports("base_velocity"):
        return ConstantTwistPolicy()
    return ConstantPolicy()


def _make_constant_twist(**_: Any) -> Policy:
    from .base import ConstantTwistPolicy

    return ConstantTwistPolicy()


def _make_replay(*, env, actions, **kwargs: Any) -> Policy:
    from .replay import ReplayPolicy

    return ReplayPolicy(env, actions, **kwargs)
