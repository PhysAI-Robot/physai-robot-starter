"""Runtime registry for robot embodiment factories.

Adding a robot is one call: build a `RobotDescriptor` bundling its factories
and pass it to `register_embodiment()`. The individual `register_*`
functions below still exist and are what `register_embodiment()` calls
internally; a robot-owned policy (e.g. a research module self-registering
"scripted" for so101) still calls `register_robot_policy()` directly, since
it registers independently of — and often after — the embodiment itself.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
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
    kind: str
    scene_defaults: SceneDefaultsFactory | None = None
    default_task: str | None = None
    env_config: EnvConfigFactory | None = None
    ros2_node: ROS2NodeFactory | None = None
    navigation: NavigationFactory | None = None
    shared_attach: SharedAttachHook | None = None
    shared_instance: SharedInstanceFactory | None = None


_FACTORIES: dict[str, RobotFactory] = {}
_KINDS: dict[str, str] = {}
_ROS2_NODE_FACTORIES: dict[str, ROS2NodeFactory] = {}
_ENV_CONFIG_FACTORIES: dict[str, EnvConfigFactory] = {}
_NAVIGATION_FACTORIES: dict[str, NavigationFactory] = {}
_SCENE_DEFAULT_FACTORIES: dict[str, SceneDefaultsFactory] = {}
_DEFAULT_TASKS: dict[str, str] = {}
_POLICY_FACTORIES: dict[tuple[str, str], RobotPolicyFactory] = {}
_SHARED_ATTACH_HOOKS: dict[str, SharedAttachHook] = {}
_SHARED_INSTANCE_FACTORIES: dict[str, SharedInstanceFactory] = {}


def register_robot(
    name: str, factory: RobotFactory, *, kind: str | None = None
) -> RobotFactory:
    """Register a robot factory under a stable configuration name."""
    if name in _FACTORIES:
        raise ValueError(f"robot {name!r} is already registered")
    _FACTORIES[name] = factory
    if kind is not None:
        _KINDS[name] = kind
    return factory


def register_embodiment(name: str, descriptor: RobotDescriptor) -> RobotDescriptor:
    """Register every factory one embodiment owns in a single call.

    This is the seam for adding a new robot: build one `RobotDescriptor` and
    call this once. It is equivalent to calling `register_robot()` plus
    whichever of `register_scene_defaults()`, `register_env_config()`,
    `register_ros2_node()`, `register_navigation()`, `register_shared_attach()`,
    and `register_shared_instance()` the descriptor supplies.
    """
    register_robot(name, descriptor.factory, kind=descriptor.kind)
    if descriptor.scene_defaults is not None:
        register_scene_defaults(name, descriptor.scene_defaults)
    if descriptor.default_task is not None:
        _DEFAULT_TASKS[name] = descriptor.default_task
    if descriptor.env_config is not None:
        register_env_config(name, descriptor.env_config)
    if descriptor.ros2_node is not None:
        register_ros2_node(name, descriptor.ros2_node)
    if descriptor.navigation is not None:
        register_navigation(name, descriptor.navigation)
    if descriptor.shared_attach is not None:
        register_shared_attach(name, descriptor.shared_attach)
    if descriptor.shared_instance is not None:
        register_shared_instance(name, descriptor.shared_instance)
    return descriptor


def robot_kind(name: str) -> str | None:
    """Return the registered embodiment kind without constructing the robot."""
    _load_builtins()
    return _KINDS.get(name)


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


def register_scene_defaults(
    name: str, factory: SceneDefaultsFactory
) -> SceneDefaultsFactory:
    """Register embodiment-owned defaults for generic scene attachment fields."""
    if name in _SCENE_DEFAULT_FACTORIES:
        raise ValueError(f"scene defaults for robot {name!r} are already registered")
    _SCENE_DEFAULT_FACTORIES[name] = factory
    return factory


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


def create_robot_policy(robot_name: str, policy_name: str, **kwargs: Any) -> Any:
    """Create a robot-owned policy without importing its implementation."""
    _load_builtins()
    try:
        factory = _POLICY_FACTORIES[(robot_name, policy_name)]
    except KeyError as exc:
        raise ValueError(f"robot {robot_name!r} has no policy {policy_name!r}") from exc
    return factory(**kwargs)


def scene_defaults(name: str) -> dict[str, Any]:
    """Return robot-owned defaults for generic scene attachment fields."""
    _load_builtins()
    factory = _SCENE_DEFAULT_FACTORIES.get(name)
    return {} if factory is None else dict(factory())


def default_task(name: str) -> str | None:
    """The task a robot runs when none is named, or None if it has no task."""
    _load_builtins()
    return _DEFAULT_TASKS.get(name)


def available_robots() -> tuple[str, ...]:
    _load_builtins()
    return tuple(sorted(_FACTORIES))


def create_robot(
    name: str, *, adapter: str = "direct_mujoco", **kwargs: Any
) -> RobotPort:
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
        raise ValueError(f"unknown ROS2 robot {name!r}; available: {choices}") from exc
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


def register_shared_attach(name: str, hook: SharedAttachHook) -> SharedAttachHook:
    """Register a hook letting a robot inject shared-world-only MJCF (e.g.
    extra cameras) into its spec at attach time."""
    if name in _SHARED_ATTACH_HOOKS:
        raise ValueError(
            f"shared-world attach hook for robot {name!r} is already registered"
        )
    _SHARED_ATTACH_HOOKS[name] = hook
    return hook


def shared_attach(name: str, spec: Any) -> None:
    """Call the robot's shared-world attach hook, if it registered one."""
    _load_builtins()
    hook = _SHARED_ATTACH_HOOKS.get(name)
    if hook is not None:
        hook(spec)


def register_shared_instance(
    name: str, factory: SharedInstanceFactory
) -> SharedInstanceFactory:
    """Register a robot-owned adapter for one binding in a shared world."""
    if name in _SHARED_INSTANCE_FACTORIES:
        raise ValueError(
            f"shared-world instance for robot {name!r} is already registered"
        )
    _SHARED_INSTANCE_FACTORIES[name] = factory
    return factory


def create_shared_instance(name: str, world: Any, config: Any) -> Any:
    """Build a robot's shared-world instance adapter through the registry."""
    _load_builtins()
    try:
        factory = _SHARED_INSTANCE_FACTORIES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(_SHARED_INSTANCE_FACTORIES)) or "none"
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
    if "so101" not in _FACTORIES:
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
    if "turtlebot4" not in _FACTORIES:
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
