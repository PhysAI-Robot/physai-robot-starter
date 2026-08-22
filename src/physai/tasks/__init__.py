"""Task definitions independent from robot embodiments and model approaches."""

from .base import Task
from .pick_place import PickPlaceTask
from .registry import available_tasks, create_task, register_task

__all__ = [
	"Task", "PickPlaceTask", "available_tasks", "create_task", "register_task",
]
