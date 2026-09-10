"""Runtime registry for robot embodiment factories."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import RobotPort

RobotFactory = Callable[..., RobotPort]
ROS2NodeFactory = Callable[..., Any]
EnvConfigFactory = Callable[..., Any]
NavigationFactory = Callable[..., Any]
_FACTORIES: dict[str, RobotFactory] = {}
_ROS2_NODE_FACTORIES: dict[str, ROS2NodeFactory] = {}
_ENV_CONFIG_FACTORIES: dict[str, EnvConfigFactory] = {}
_NAVIGATION_FACTORIES: dict[str, NavigationFactory] = {}


def register_robot(name: str, factory: RobotFactory) -> RobotFactory:
    """Register a robot factory under a stable configuration name."""
    if name in _FACTORIES:
        raise ValueError(f"robot {name!r} is already registered")
    _FACTORIES[name] = factory
    return factory


def register_ros2_node(name: str, factory: ROS2NodeFactory) -> ROS2NodeFactory:
    """Register an embodiment-specific ROS2 node factory."""
    if name in _ROS2_NODE_FACTORIES:
        raise ValueError(f"ROS2 node for robot {name!r} is already registered")
    _ROS2_NODE_FACTORIES[name] = factory
    return factory


def register_env_config(name: str, factory: EnvConfigFactory) -> EnvConfigFactory:
    """Register a robot-owned environment configuration factory."""
    if name in _ENV_CONFIG_FACTORIES:
        raise ValueError(f"environment config for robot {name!r} is already registered")
    _ENV_CONFIG_FACTORIES[name] = factory
    return factory


def register_navigation(name: str, factory: NavigationFactory) -> NavigationFactory:
    """Register a robot-owned deterministic navigation baseline."""
    if name in _NAVIGATION_FACTORIES:
        raise ValueError(f"navigation for robot {name!r} is already registered")
    _NAVIGATION_FACTORIES[name] = factory
    return factory


def available_robots() -> tuple[str, ...]:
    _load_builtins()
    return tuple(sorted(_FACTORIES))


def create_robot(name: str, *, adapter: str = "direct_mujoco", **kwargs: Any) -> RobotPort:
    """Create a robot through the selected simulation or hardware adapter."""
    _load_builtins()
    try:
        factory = _FACTORIES[name]
    except KeyError as exc:
        choices = ", ".join(available_robots())
        raise ValueError(f"unknown robot {name!r}; available: {choices}") from exc
    return factory(adapter=adapter, **kwargs)


def create_ros2_node(name: str, node: Any, **kwargs: Any) -> Any:
    """Create an embodiment ROS2 node through the shared registry."""
    _load_builtins()
    try:
        factory = _ROS2_NODE_FACTORIES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(_ROS2_NODE_FACTORIES))
        raise ValueError(
            f"unknown ROS2 robot {name!r}; available: {choices}"
        ) from exc
    return factory(node, **kwargs)


def create_env_config(name: str, **kwargs: Any) -> Any:
    """Create an embodiment-owned environment config through the registry."""
    _load_builtins()
    try:
        factory = _ENV_CONFIG_FACTORIES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(_ENV_CONFIG_FACTORIES))
        raise ValueError(f"unknown robot {name!r}; available: {choices}") from exc
    return factory(**kwargs)


def navigate(name: str, **kwargs: Any) -> Any:
    """Run a robot's registered navigation baseline."""
    _load_builtins()
    try:
        factory = _NAVIGATION_FACTORIES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(_NAVIGATION_FACTORIES)) or "none"
        raise ValueError(
            f"robot {name!r} has no registered navigation baseline; available: {choices}"
        ) from exc
    return factory(**kwargs)


def available_ros2_robots() -> tuple[str, ...]:
    """Return robots with a registered ROS2 node adapter."""
    _load_builtins()
    return tuple(sorted(_ROS2_NODE_FACTORIES))


def _load_builtins() -> None:
    if "so101" not in _FACTORIES:
        from .so101.factory import make_so101

        register_robot("so101", make_so101)
    if "so101" not in _ENV_CONFIG_FACTORIES:
        from .so101.env import EnvConfig

        register_env_config("so101", EnvConfig)
    if "turtlebot4" not in _FACTORIES:
        from .turtlebot.factory import make_turtlebot4

        register_robot("turtlebot4", make_turtlebot4)
    if "so101" not in _ROS2_NODE_FACTORIES:
        from .so101.ros2_node import SO101ROS2Node

        register_ros2_node("so101", SO101ROS2Node)
    if "turtlebot4" not in _ROS2_NODE_FACTORIES:
        from .turtlebot.ros2_node import TurtleBot4ROS2Node

        register_ros2_node("turtlebot4", TurtleBot4ROS2Node)
    if "turtlebot4" not in _ENV_CONFIG_FACTORIES:
        from .turtlebot.env import TurtleBot4Config

        register_env_config("turtlebot4", TurtleBot4Config)
    if "turtlebot4" not in _NAVIGATION_FACTORIES:
        from .turtlebot.navigation import navigate_to_coordinates

        register_navigation("turtlebot4", navigate_to_coordinates)