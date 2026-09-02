"""Robot embodiment contracts, adapters, and built-in factories."""

from .adapters import ADAPTER_NAMES, DirectMuJoCoAdapter, select_adapter
from .base import KinematicsPort, RobotEnv, RobotPort, RobotSpec
from .registry import available_robots, create_robot, register_robot

__all__ = [
    "RobotEnv",
    "RobotPort",
    "RobotSpec",
    "KinematicsPort",
    "DirectMuJoCoAdapter",
    "ADAPTER_NAMES",
    "select_adapter",
    "available_robots",
    "create_robot",
    "register_robot",
]