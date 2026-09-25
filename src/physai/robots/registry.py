"""Runtime registry for robot embodiment factories.

Adding a robot is one call: build a `RobotDescriptor` bundling its factories
and pass it to `register_embodiment()`, which stores that one descriptor. The
individual `register_*` functions below fill in one optional field of an
already registered robot; a robot-owned policy (e.g. a research module
self-registering "scripted" for so101) still calls `register_robot_policy()`
directly, since it registers independently of — and often after — the
embodiment itself.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from .base import RobotPort

RobotFactory = Callable[..., RobotPort]
ROS2NodeFactory = Callable[..., Any]
EnvConfigFactory = Callable[..., Any]
NavigationFactory = Callable[..., Any]
SceneDefaultsFactory = Callable[[], dict[str, Any]]
RobotPolicyFactory = Callable[..., Any]
SharedAttachHook = Callable[[Any], None]
SharedInstanceFactory = Callable[[Any, Any], Any]


@dataclass(frozen=True)
class RobotDescriptor:
    """Everything one embodiment registers, bundled for one `register_embodiment()` call."""

    factory: RobotFactory
    kind: str | None
    scene_defaults: SceneDefaultsFactory | None = None
    default_task: str | None = None
    env_config: EnvConfigFactory | None = None
    ros2_node: ROS2NodeFactory | None = None
    navigation: NavigationFactory | None = None
    shared_attach: SharedAttachHook | None = None
    shared_instance: SharedInstanceFactory | None = None


_ROBOTS: dict[str, RobotDescriptor] = {}
_POLICY_FACTORIES: dict[tuple[str, str], RobotPolicyFactory] = {}


def register_embodiment(name: str, descriptor: RobotDescriptor) -> RobotDescriptor:
    """Register every factory one embodiment owns in a single call.

    This is the seam for adding a new robot: build one `RobotDescriptor` and
    call this once.
    """
    if name in _ROBOTS:
        raise ValueError(f"robot {name!r} is already registered")
    _ROBOTS[name] = descriptor
    return descriptor


def register_robot(
    name: str, factory: RobotFactory, *, kind: str | None = None
) -> RobotFactory:
    """Register a robot factory under a stable configuration name."""
    register_embodiment(name, RobotDescriptor(factory=factory, kind=kind))
    return factory


def _extend(name: str, field: str, label: str, value: Any) -> Any:
    """Set one still-empty optional field of an already registered robot."""
    try:
        descriptor = _ROBOTS[name]
    except KeyError:
        raise ValueError(f"robot {name!r} is not registered") from None
    if getattr(descriptor, field) is not None:
        raise ValueError(f"{label} for robot {name!r} is already registered")
    _ROBOTS[name] = replace(descriptor, **{field: value})
    return value


def register_ros2_node(name: str, factory: ROS2NodeFactory) -> ROS2NodeFactory:
    """Register an embodiment-specific ROS2 node factory."""
    return _extend(name, "ros2_node", "ROS2 node", factory)


def register_env_config(name: str, factory: EnvConfigFactory) -> EnvConfigFactory:
    """Register a robot-owned environment configuration factory."""
    return _extend(name, "env_config", "environment config", factory)


def register_navigation(name: str, factory: NavigationFactory) -> NavigationFactory:
    """Register a robot-owned deterministic navigation baseline."""
    return _extend(name, "navigation", "navigation", factory)


def register_scene_defaults(
    name: str, factory: SceneDefaultsFactory
) -> SceneDefaultsFactory:
    """Register embodiment-owned defaults for generic scene attachment fields."""
    return _extend(name, "scene_defaults", "scene defaults", factory)


def register_shared_attach(name: str, hook: SharedAttachHook) -> SharedAttachHook:
    """Register a hook letting a robot inject shared-world-only MJCF (e.g.
    extra cameras) into its spec at attach time."""
    return _extend(name, "shared_attach", "shared-world attach hook", hook)


def register_shared_instance(
    name: str, factory: SharedInstanceFactory
) -> SharedInstanceFactory:
    """Register a robot-owned adapter for one binding in a shared world."""
    return _extend(name, "shared_instance", "shared-world instance", factory)


def register_robot_policy(
    robot_name: str, policy_name: str, factory: RobotPolicyFactory
) -> RobotPolicyFactory:
    """Register a policy implementation owned by one robot adapter."""
    key = (robot_name, policy_name)
    if key in _POLICY_FACTORIES:
        raise ValueError(
            f"policy {policy_name!r} for robot {robot_name!r} is already registered"
        )
    _POLICY_FACTORIES[key] = factory
    return factory


def _having(field: str) -> dict[str, Any]:
    """Robot name -> the value of one optional descriptor field, where it is set."""
    _load_builtins()
    return {
        name: getattr(descriptor, field)
        for name, descriptor in _ROBOTS.items()
        if getattr(descriptor, field) is not None
    }


def robot_kind(name: str) -> str | None:
    """Return the registered embodiment kind without constructing the robot."""
    _load_builtins()
    descriptor = _ROBOTS.get(name)
    return None if descriptor is None else descriptor.kind


def scene_defaults(name: str) -> dict[str, Any]:
    """Return robot-owned defaults for generic scene attachment fields."""
    factory = _having("scene_defaults").get(name)
    return {} if factory is None else dict(factory())


def default_task(name: str) -> str | None:
    """The task a robot runs when none is named, or None if it has no task."""
    return _having("default_task").get(name)


def available_robots() -> tuple[str, ...]:
    _load_builtins()
    return tuple(sorted(_ROBOTS))


def available_ros2_robots() -> tuple[str, ...]:
    """Return robots with a registered ROS2 node adapter."""
    return tuple(sorted(_having("ros2_node")))


def create_robot(
    name: str, *, adapter: str = "direct_mujoco", **kwargs: Any
) -> RobotPort:
    """Create a robot through the selected simulation or hardware adapter."""
    _load_builtins()
    try:
        descriptor = _ROBOTS[name]
    except KeyError as exc:
        choices = ", ".join(available_robots())
        raise ValueError(f"unknown robot {name!r}; available: {choices}") from exc
    return descriptor.factory(adapter=adapter, **kwargs)


def create_robot_policy(robot_name: str, policy_name: str, **kwargs: Any) -> Any:
    """Create a robot-owned policy without importing its implementation."""
    _load_builtins()
    try:
        factory = _POLICY_FACTORIES[(robot_name, policy_name)]
    except KeyError as exc:
        raise ValueError(f"robot {robot_name!r} has no policy {policy_name!r}") from exc
    return factory(**kwargs)


def create_ros2_node(name: str, node: Any, **kwargs: Any) -> Any:
    """Create an embodiment ROS2 node through the shared registry."""
    factories = _having("ros2_node")
    try:
        factory = factories[name]
    except KeyError as exc:
        choices = ", ".join(sorted(factories))
        raise ValueError(f"unknown ROS2 robot {name!r}; available: {choices}") from exc
    return factory(node, **kwargs)


def create_env_config(name: str, **kwargs: Any) -> Any:
    """Create an embodiment-owned environment config through the registry."""
    factories = _having("env_config")
    try:
        factory = factories[name]
    except KeyError as exc:
        choices = ", ".join(sorted(factories))
        raise ValueError(f"unknown robot {name!r}; available: {choices}") from exc
    return factory(**kwargs)


def navigate(name: str, **kwargs: Any) -> Any:
    """Run a robot's registered navigation baseline."""
    factories = _having("navigation")
    try:
        factory = factories[name]
    except KeyError as exc:
        choices = ", ".join(sorted(factories)) or "none"
        raise ValueError(
            f"robot {name!r} has no registered navigation baseline; available: {choices}"
        ) from exc
    return factory(**kwargs)


def shared_attach(name: str, spec: Any) -> None:
    """Call the robot's shared-world attach hook, if it registered one."""
    hook = _having("shared_attach").get(name)
    if hook is not None:
        hook(spec)


def create_shared_instance(name: str, world: Any, config: Any) -> Any:
    """Build a robot's shared-world instance adapter through the registry."""
    factories = _having("shared_instance")
    try:
        factory = factories[name]
    except KeyError as exc:
        choices = ", ".join(sorted(factories)) or "none"
        raise ValueError(
            f"robot {name!r} has no registered shared-world instance; "
            f"available: {choices}"
        ) from exc
    return factory(world, config)


def _load_builtins() -> None:
    # so101's "scripted" and "visual_servo" policies are research modules
    # (research/scripted_experts, research/classical_control); they register
    # themselves with register_robot_policy() on import. This registry never
    # imports them directly (core must not import research/).
    if "so101" not in _ROBOTS:
        from .so101.env import EnvConfig
        from .so101.factory import make_so101
        from .so101.ros2_node import SO101ROS2Node
        from .so101.scene import scene_defaults as so101_scene_defaults
        from .so101.shared import SO101SharedInstance, so101_shared_attach

        register_embodiment(
            "so101",
            RobotDescriptor(
                factory=make_so101,
                kind="fixed_base_manipulator",
                scene_defaults=so101_scene_defaults,
                default_task="pick_place",
                env_config=EnvConfig,
                ros2_node=SO101ROS2Node,
                shared_attach=so101_shared_attach,
                shared_instance=SO101SharedInstance,
            ),
        )
    if "turtlebot4" not in _ROBOTS:
        from .turtlebot.env import TurtleBot4Config
        from .turtlebot.factory import make_turtlebot4
        from .turtlebot.navigation import navigate_to_coordinates
        from .turtlebot.ros2_node import TurtleBot4ROS2Node
        from .turtlebot.shared import TurtleBot4SharedInstance

        register_embodiment(
            "turtlebot4",
            RobotDescriptor(
                factory=make_turtlebot4,
                kind="mobile_base",
                env_config=TurtleBot4Config,
                ros2_node=TurtleBot4ROS2Node,
                navigation=navigate_to_coordinates,
                shared_instance=TurtleBot4SharedInstance,
            ),
        )
