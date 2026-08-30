"""Task definitions independent from robot embodiments and model approaches."""

from .base import Task, TaskBackend
from .pick_place_minimal import PickPlaceBackend, PickPlaceTask
from .registry import available_tasks, create_task, register_task
from .sorting_minimal import SortingBackend, SortingTask

__all__ = [
	"Task", "TaskBackend", "PickPlaceBackend", "PickPlaceTask",
	"SortingBackend", "SortingTask", "available_tasks", "create_task",
	"register_task",
]
