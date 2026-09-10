"""Robot embodiment contracts, adapters, and built-in factories."""

from .adapters import ADAPTER_NAMES, DirectMuJoCoAdapter, select_adapter
from .base import KinematicsPort, RobotEnv, RobotPort, RobotSpec
from .kinematics_registry import (
    available_kinematics_adapters,
    create_kinematics_adapter,
    register_kinematics_adapter,
)
from .registry import (
    available_robots,
    available_ros2_robots,
    create_robot,
    create_ros2_node,
    create_env_config,
    navigate,
    register_robot,
    register_env_config,
    register_navigation,
    register_ros2_node,
)

__all__ = [
    "RobotEnv",
    "RobotPort",
    "RobotSpec",
    "KinematicsPort",
    "available_kinematics_adapters",
    "create_kinematics_adapter",
    "register_kinematics_adapter",
    "DirectMuJoCoAdapter",
    "ADAPTER_NAMES",
    "select_adapter",
    "available_robots",
    "available_ros2_robots",
    "create_robot",
    "create_ros2_node",
    "create_env_config",
    "navigate",
    "register_robot",
    "register_env_config",
    "register_navigation",
    "register_ros2_node",
]