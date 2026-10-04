"""Task definitions independent from robot embodiments and model approaches."""

from .base import Task, TaskBackend
from .single_cube_fixed_place import (
    SingleCubeFixedPlaceBackend,
    SingleCubeFixedPlaceTask,
)
from .registry import available_tasks, create_task, register_task
from .runtime import TaskRuntime
from .sorting_minimal import SortingBackend, SortingTask

__all__ = [
    "SingleCubeFixedPlaceBackend",
    "SingleCubeFixedPlaceTask",
    "SortingBackend",
    "SortingTask",
    "Task",
    "TaskBackend",
    "TaskRuntime",
    "available_tasks",
    "create_task",
    "register_task",
]
