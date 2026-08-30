"""Load typed runtime configuration from YAML files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .robots.so101.env import EnvConfig
from .sim.scenes import create_scene, get_scene_definition


@dataclass(frozen=True)
class TaskConfig:
    """Typed robot, task, scene, and environment configuration."""

    robot: str
    task: str
    scene_name: str
    env: EnvConfig


_SCENE_TUPLE_FIELDS = {
    "table_pos",
    "table_size",
    "target_pos",
    "cube_names",
    "cube_rgba",
}
_ENV_TUPLE_FIELDS = {"cameras", "cube_x_range", "cube_y_range"}


def load_task_config(path: str | Path) -> TaskConfig:
    """Load a task YAML file and construct its typed scene and env configs."""
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"configuration root must be a mapping: {config_path}")

    robot = _required_string(data, "robot")
    task_data = _required_mapping(data, "task")
    task_name = _required_string(task_data, "name")
    scene_name = _required_string(task_data, "scene")
    scene_definition = get_scene_definition(scene_name)
    if task_name not in scene_definition.task_names:
        supported = ", ".join(scene_definition.task_names)
        raise ValueError(
            f"scene {scene_name!r} is incompatible with task {task_name!r}; "
            f"supported tasks: {supported}"
        )

    scene_data = dict(_required_mapping(data, "scene"))
    if "robot_xml" in scene_data:
        scene_data["robot_xml"] = _resolve_path(scene_data["robot_xml"], config_path)
    _convert_lists_to_tuples(scene_data, _SCENE_TUPLE_FIELDS)
    scene = create_scene(scene_name, **scene_data)

    env_data = dict(_required_mapping(data, "env"))
    configured_task = env_data.pop("task", task_name)
    if configured_task != task_name:
        raise ValueError(
            f"task mismatch: task.name={task_name!r}, env.task={configured_task!r}"
        )
    _convert_lists_to_tuples(env_data, _ENV_TUPLE_FIELDS)
    env = EnvConfig(scene=scene, task=task_name, **env_data)
    return TaskConfig(
        robot=robot,
        task=task_name,
        scene_name=scene_name,
        env=env,
    )


def _required_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"configuration field {key!r} must be a mapping")
    return value


def _required_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"configuration field {key!r} must be a non-empty string")
    return value


def _resolve_path(value: Any, config_path: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("scene.robot_xml must be a non-empty path")
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate.resolve()
    return (config_path.parent.parent / candidate).resolve()


def _convert_lists_to_tuples(data: dict[str, Any], keys: set[str]) -> None:
    for key in keys:
        if isinstance(data.get(key), list):
            data[key] = tuple(data[key])


__all__ = ["TaskConfig", "load_task_config"]