"""Robot embodiment contracts, adapters, and built-in factories."""

from .adapters import (
    DirectAdapter,
    available_adapters,
    create_adapter,
    register_adapter,
)
from .base import KinematicsPort, RobotPort, RobotSpec, RobotTrainingContract
from .registry import (
    RobotDescriptor,
    available_robots,
    available_ros2_robots,
    create_env_config,
    create_jog_resolver,
    create_robot,
    create_robot_policy,
    create_ros2_node,
    create_shared_instance,
    default_task,
    has_robot_policy,
    navigate,
    register_embodiment,
    register_robot_policy,
    shared_attach,
)

__all__ = [
    "DirectAdapter",
    "KinematicsPort",
    "RobotDescriptor",
    "RobotPort",
    "RobotSpec",
    "RobotTrainingContract",
    "available_adapters",
    "available_robots",
    "available_ros2_robots",
    "create_adapter",
    "create_env_config",
    "create_jog_resolver",
    "create_robot",
    "create_robot_policy",
    "create_ros2_node",
    "create_shared_instance",
    "default_task",
    "navigate",
    "register_adapter",
    "register_embodiment",
    "has_robot_policy",
    "register_robot_policy",
    "shared_attach",
]
