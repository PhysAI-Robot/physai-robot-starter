"""Robot embodiment contracts, adapters, and built-in factories."""

from .adapters import DirectMuJoCoAdapter, select_adapter
from .base import KinematicsPort, RobotPort, RobotSpec, RobotTrainingContract
from .registry import (
    available_robots,
    available_ros2_robots,
    create_env_config,
    create_robot,
    create_robot_policy,
    create_ros2_node,
    navigate,
    register_env_config,
    register_navigation,
    register_robot,
    register_robot_policy,
    register_ros2_node,
)

__all__ = [
    "DirectMuJoCoAdapter",
    "KinematicsPort",
    "RobotPort",
    "RobotSpec",
    "RobotTrainingContract",
    "available_robots",
    "available_ros2_robots",
    "create_env_config",
    "create_robot",
    "create_robot_policy",
    "create_ros2_node",
    "navigate",
    "register_env_config",
    "register_navigation",
    "register_robot",
    "register_robot_policy",
    "register_ros2_node",
    "select_adapter",
]
