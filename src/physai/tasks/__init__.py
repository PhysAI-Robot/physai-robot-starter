"""Task definitions independent from robot embodiments and model approaches."""

from .base import Task
from .single_cube_place import (
    SingleCubePlaceBackend,
    SingleCubePlaceTask,
)
from .registry import available_tasks, create_task, register_task
from .runtime import TaskRuntime
from .sorting_minimal import SortingBackend, SortingTask

__all__ = [
    "SingleCubePlaceBackend",
    "SingleCubePlaceTask",
    "SortingBackend",
    "SortingTask",
    "Task",
    "TaskRuntime",
    "available_tasks",
    "create_task",
    "register_task",
]
