"""Robot embodiment contracts, adapters, and built-in factories."""

from .adapters import ADAPTER_NAMES, DirectMuJoCoAdapter, select_adapter
from .base import KinematicsPort, RobotEnv, RobotPort, RobotSpec, RobotTrainingContract
from .kinematics_registry import (
    available_kinematics_adapters,
    create_kinematics_adapter,
    register_kinematics_adapter,
)
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
    "ADAPTER_NAMES",
    "DirectMuJoCoAdapter",
    "KinematicsPort",
    "RobotEnv",
    "RobotPort",
    "RobotSpec",
    "RobotTrainingContract",
    "available_kinematics_adapters",
    "available_robots",
    "available_ros2_robots",
    "create_env_config",
    "create_kinematics_adapter",
    "create_robot",
    "create_robot_policy",
    "create_ros2_node",
    "navigate",
    "register_env_config",
    "register_kinematics_adapter",
    "register_navigation",
    "register_robot",
    "register_robot_policy",
    "register_ros2_node",
    "select_adapter",
]
