"""Load typed runtime configuration from YAML files.

This is the pre-manifest configuration surface: one robot's task+scene+env
(`load_task_config`), a heterogeneous world's robot placement
(`load_world_config`), and shared simulation defaults (`load_sim_config`).
It stays for CLI back-compat; `physai.config.manifest` is the new, unified
session-manifest schema (robots + scene + task + policy + backend +
viewer in one file).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..robots import create_env_config
from ..sim.domain_randomization import DomainRandomizationConfig
from ..sim.scenes import create_scene, get_scene_definition
from ..sim.world import RobotInstanceConfig


@dataclass(frozen=True)
class SimulationConfig:
    """Settings shared by robot environments and simulation entry points."""

    seed: int = 0
    domain_randomization: DomainRandomizationConfig = field(
        default_factory=DomainRandomizationConfig
    )


@dataclass(frozen=True)
class TaskConfig:
    """Typed robot, task, scene, and environment configuration."""

    robot: str
    task: str
    scene_name: str
    env: Any
    success_xy_tol: float = 0.04
    success_hold_steps: int = 10
    simulation: SimulationConfig = field(default_factory=SimulationConfig)


@dataclass(frozen=True)
class WorldConfig:
    """Typed configuration for a heterogeneous shared MuJoCo world."""

    instances: tuple[RobotInstanceConfig, ...]
    timestep: float = 0.002
    control_hz: float = 30.0
    add_floor: bool = True


_SCENE_TUPLE_FIELDS = {
    "table_pos",
    "table_size",
    "target_pos",
    "cube_names",
    "cube_rgba",
}
_ENV_TUPLE_FIELDS = {"cameras", "cube_x_range", "cube_y_range"}


def load_sim_config(path: str | Path) -> SimulationConfig:
    """Load shared simulation settings from a YAML mapping."""
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(  # noqa: TRY004
            f"configuration root must be a mapping: {config_path}"
        )

    return _parse_simulation_config(data, config_path)


def _parse_simulation_config(
    data: dict[str, Any],
    source: Path | str,
) -> SimulationConfig:
    seed = data.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(  # noqa: TRY004
            f"{source}: simulation seed must be an integer"
        )
    randomization_data = data.get("domain_randomization", {})
    if not isinstance(randomization_data, dict):
        raise ValueError(  # noqa: TRY004
            f"{source}: domain_randomization must be a mapping"
        )
    enabled = randomization_data.get("enabled", False)
    kwargs: dict[str, Any] = {"enabled": enabled}
    for key in (
        "friction_scale",
        "mass_scale",
        "lighting_scale",
        "clutter_x_range",
        "clutter_y_range",
    ):
        if key in randomization_data:
            value = randomization_data[key]
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(f"{source}: domain_randomization.{key} must be a pair")
            kwargs[key] = tuple(float(item) for item in value)
    if "camera_position_jitter" in randomization_data:
        kwargs["camera_position_jitter"] = float(
            randomization_data["camera_position_jitter"]
        )
    if "clutter_clearance" in randomization_data:
        kwargs["clutter_clearance"] = float(randomization_data["clutter_clearance"])
    return SimulationConfig(
        seed=seed,
        domain_randomization=DomainRandomizationConfig(**kwargs),
    )


def load_task_config(path: str | Path) -> TaskConfig:
    """Load a task YAML file and construct its typed scene and env configs."""
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(  # noqa: TRY004
            f"configuration root must be a mapping: {config_path}"
        )

    simulation_data = data.get("simulation", {})
    if not isinstance(simulation_data, dict):
        raise ValueError(  # noqa: TRY004
            "configuration field 'simulation' must be a mapping"
        )
    simulation = _parse_simulation_config(simulation_data, config_path)

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
    success_xy_tol = env_data.pop("success_xy_tol", 0.04)
    success_hold_steps = env_data.pop("success_hold_steps", 10)
    _convert_lists_to_tuples(env_data, _ENV_TUPLE_FIELDS)
    env = create_env_config(robot, scene=scene, **env_data)
    return TaskConfig(
        robot=robot,
        task=task_name,
        scene_name=scene_name,
        env=env,
        simulation=simulation,
        success_xy_tol=success_xy_tol,
        success_hold_steps=success_hold_steps,
    )


def load_world_config(path: str | Path) -> WorldConfig:
    """Load a shared-world manifest with namespaced robot instances."""
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(  # noqa: TRY004
            f"configuration root must be a mapping: {config_path}"
        )

    raw_instances = data.get("robots")
    if not isinstance(raw_instances, list) or not raw_instances:
        raise ValueError("world configuration field 'robots' must be a non-empty list")
    instances = []
    for index, raw_instance in enumerate(raw_instances):
        if not isinstance(raw_instance, dict):
            raise ValueError(  # noqa: TRY004
                f"world robots[{index}] must be a mapping"
            )
        instance_id = _required_string(raw_instance, "id")
        robot_name = _required_string(raw_instance, "robot")
        model_value = raw_instance.get("model")
        model_path = _resolve_path(model_value, config_path)
        position = _vector(raw_instance.get("position", (0.0, 0.0, 0.0)), 3, "position")
        quaternion = _vector(
            raw_instance.get("quaternion", (1.0, 0.0, 0.0, 0.0)),
            4,
            "quaternion",
        )
        instances.append(
            RobotInstanceConfig(
                instance_id=instance_id,
                model_path=model_path,
                robot_name=robot_name,
                position=position,
                quaternion=quaternion,
            )
        )
    timestep = float(data.get("timestep", 0.002))
    control_hz = float(data.get("control_hz", 30.0))
    add_floor = data.get("add_floor", True)
    if timestep <= 0 or control_hz <= 0:
        raise ValueError("world timestep and control_hz must be positive")
    if not isinstance(add_floor, bool):
        raise ValueError(  # noqa: TRY004
            "world add_floor must be a boolean"
        )
    return WorldConfig(
        instances=tuple(instances),
        timestep=timestep,
        control_hz=control_hz,
        add_floor=add_floor,
    )


def _required_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(  # noqa: TRY004
            f"configuration field {key!r} must be a mapping"
        )
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


def _vector(value: Any, size: int, field_name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"world robot {field_name} must contain {size} values")
    return tuple(float(item) for item in value)


def _convert_lists_to_tuples(data: dict[str, Any], keys: set[str]) -> None:
    for key in keys:
        if isinstance(data.get(key), list):
            data[key] = tuple(data[key])


__all__ = [
    "DomainRandomizationConfig",
    "SimulationConfig",
    "TaskConfig",
    "WorldConfig",
    "load_sim_config",
    "load_task_config",
    "load_world_config",
]
