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


def available_policies() -> tuple[str, ...]:
    """Return policy names available to the current runtime."""
    _load_builtins()
    return tuple(sorted(_FACTORIES))


def create_policy(name: str, **kwargs: Any) -> Policy:
    """Create a policy without exposing its concrete implementation."""
    _load_builtins()
    try:
        factory = _FACTORIES[name]
    except KeyError as exc:
        choices = ", ".join(available_policies())
        raise ValueError(f"unknown policy {name!r}; available: {choices}") from exc
    return factory(**kwargs)


def _load_builtins() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return

    register_policy("constant", _make_constant)
    register_policy("constant_twist", _make_constant_twist)
    register_policy("scripted", _make_scripted)
    register_policy("replay", _make_replay)
    register_policy("lerobot", _make_lerobot)
    _BUILTINS_LOADED = True


def _make_constant(**_: Any) -> Policy:
    from .base import ConstantPolicy

    return ConstantPolicy()


def _make_constant_twist(**_: Any) -> Policy:
    from .base import ConstantTwistPolicy

    return ConstantTwistPolicy()


def _make_scripted(*, env, **kwargs: Any) -> Policy:
    from .scripted import ScriptedPickPlace

    return ScriptedPickPlace(env.kin, env, cfg=kwargs.get("cfg"))


def _make_replay(*, env, actions, **kwargs: Any) -> Policy:
    from .vla_adapter import ReplayPolicy

    return ReplayPolicy(env, actions, **kwargs)


def _make_lerobot(*, env, checkpoint, **kwargs: Any) -> Policy:
    from .vla_adapter import LeRobotPolicy
    return LeRobotPolicy.from_checkpoint(env, checkpoint, **kwargs)