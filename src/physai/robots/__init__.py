"""Robot embodiment contracts, adapters, and built-in factories."""

from .adapters import (
    DirectMuJoCoAdapter,
    available_adapters,
    create_adapter,
    register_adapter,
    select_adapter,
)
from .base import KinematicsPort, RobotPort, RobotSpec, RobotTrainingContract
from .registry import (
    RobotDescriptor,
    available_robots,
    available_ros2_robots,
    create_env_config,
    create_robot,
    create_robot_policy,
    create_ros2_node,
    navigate,
    register_embodiment,
    register_env_config,
    register_navigation,
    register_robot,
    register_robot_policy,
    register_ros2_node,
)

__all__ = [
    "DirectMuJoCoAdapter",
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
    "create_robot",
    "create_robot_policy",
    "create_ros2_node",
    "navigate",
    "register_adapter",
    "register_embodiment",
    "register_env_config",
    "register_navigation",
    "register_robot",
    "register_robot_policy",
    "register_ros2_node",
    "select_adapter",
]
