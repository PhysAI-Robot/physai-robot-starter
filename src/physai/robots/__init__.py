"""Robot embodiment contracts, adapters, and built-in factories."""

from .adapters import ADAPTER_NAMES, DirectMuJoCoAdapter, select_adapter
from .base import KinematicsPort, RobotEnv, RobotPort, RobotSpec
from .registry import (
    available_robots,
    available_ros2_robots,
    create_robot,
    create_ros2_node,
    register_robot,
    register_ros2_node,
)

__all__ = [
    "RobotEnv",
    "RobotPort",
    "RobotSpec",
    "KinematicsPort",
    "DirectMuJoCoAdapter",
    "ADAPTER_NAMES",
    "select_adapter",
    "available_robots",
    "available_ros2_robots",
    "create_robot",
    "create_ros2_node",
    "register_robot",
    "register_ros2_node",
]