"""Robot embodiment contracts and built-in robot factories."""

from .base import RobotEnv, RobotSpec
from .registry import available_robots, create_robot, register_robot

__all__ = [
    "RobotEnv",
    "RobotSpec",
    "available_robots",
    "create_robot",
    "register_robot",
]