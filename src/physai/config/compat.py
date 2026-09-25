"""Turn the older run configuration into a session manifest.

`configs/tasks/<robot>/*.yaml` (one robot, task, scene, and env),
`configs/worlds/*.yaml` (robot placement in a shared world), and a bare robot
name are the inputs `scripts/run_sim.py` accepted before manifests existed.
Each converter builds manifest-shaped data and runs it through
`parse_manifest`, so a legacy file gets exactly the validation a manifest
does and the rest of the code only ever sees a `SessionManifest`.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from ..robots.registry import default_task
from .legacy import SimulationConfig, _required_mapping, _required_string
from .manifest import SCHEMA_VERSION, SessionManifest, parse_manifest

# The camera resolution a bare `--robot` run has always rendered at; scene
# files that name a task without one use the scene's own (smaller) default.
BARE_ROBOT_CAMERA_SIZE = (640, 480)
# Episode length of a run that names neither a config nor --max-steps.
BARE_ROBOT_MAX_STEPS = 600


def manifest_from_task_file(
    path: str | Path, *, simulation: SimulationConfig
) -> SessionManifest:
    """A `configs/tasks/<robot>/<task>.yaml` file as a one-robot manifest."""
    source = Path(path).resolve()
    data = _read_mapping(source)
    robot = _required_string(data, "robot")
    task_data = _required_mapping(data, "task")
    task_name = _required_string(task_data, "name")
    env_data = dict(_required_mapping(data, "env"))
    configured_task = env_data.pop("task", task_name)
    if configured_task != task_name:
        raise ValueError(
            f"task mismatch: task.name={task_name!r}, env.task={configured_task!r}"
        )
    task_kwargs = {}
    if "success_xy_tol" in env_data:
        task_kwargs["success_xy_tol"] = env_data.pop("success_xy_tol")
    seed = env_data.pop("seed", None)
    hold_steps = env_data.pop("success_hold_steps", None)

    manifest_data: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "scene": {
            "name": _required_string(task_data, "scene"),
            "overrides": dict(_required_mapping(data, "scene")),
        },
        "robots": [{"id": robot, "robot": robot, "config": env_data}],
        "task": task_name,
        "task_kwargs": task_kwargs,
    }
    if hold_steps is not None:
        manifest_data["success_hold_steps"] = hold_steps
    return _with_simulation(parse_manifest(manifest_data, source), simulation, seed)


def manifest_from_world_file(
    path: str | Path, *, simulation: SimulationConfig
) -> SessionManifest:
    """A `configs/worlds/*.yaml` file as a shared-world manifest."""
    source = Path(path).resolve()
    data = _read_mapping(source)
    raw_robots = data.get("robots")
    if not isinstance(raw_robots, list) or not raw_robots:
        raise ValueError("world configuration field 'robots' must be a non-empty list")
    robots = []
    for index, raw in enumerate(raw_robots):
        if not isinstance(raw, dict):
            raise ValueError(f"world robots[{index}] must be a mapping")  # noqa: TRY004
        robot: dict[str, Any] = {
            "id": raw.get("id"),
            "robot": raw.get("robot"),
            "model": raw.get("model"),
            "pose": {},
        }
        for key in ("position", "quaternion"):
            if key in raw:
                robot["pose"][key] = raw[key]
        robots.append(robot)
    manifest_data = {
        "schema_version": SCHEMA_VERSION,
        "world": {
            key: data[key]
            for key in ("timestep", "control_hz", "add_floor")
            if key in data
        },
        "robots": robots,
    }
    return _with_simulation(parse_manifest(manifest_data, source), simulation, None)


def manifest_for_robot(robot: str, *, simulation: SimulationConfig) -> SessionManifest:
    """The manifest of `run_sim.py --robot <name>` without a config file."""
    task = default_task(robot)
    manifest_data: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "robots": [
            {
                "id": robot,
                "robot": robot,
                "config": {"max_steps": BARE_ROBOT_MAX_STEPS},
            }
        ],
    }
    if task is not None:
        width, height = BARE_ROBOT_CAMERA_SIZE
        manifest_data["task"] = task
        manifest_data["scene"] = {
            "overrides": {"camera_width": width, "camera_height": height}
        }
    return _with_simulation(
        parse_manifest(manifest_data, Path(f"<--robot {robot}>")), simulation, None
    )


def with_overrides(
    manifest: SessionManifest,
    *,
    seed: int | None = None,
    max_steps: int | None = None,
    camera_size: int | None = None,
    policy: str | None = None,
) -> SessionManifest:
    """The manifest with command-line overrides applied; ``None`` keeps a value."""
    changes: dict[str, Any] = {}
    if seed is not None:
        changes["simulation"] = replace(manifest.simulation, seed=seed)
    if camera_size is not None:
        changes["scene"] = replace(
            manifest.scene,
            overrides={
                **manifest.scene.overrides,
                "camera_width": camera_size,
                "camera_height": camera_size,
            },
        )
    if manifest.world is None and (max_steps is not None or policy is not None):
        changes["robots"] = tuple(
            replace(
                robot,
                config=(
                    robot.config
                    if max_steps is None
                    else {**robot.config, "max_steps": max_steps}
                ),
                policy=robot.policy if policy is None else policy,
            )
            for robot in manifest.robots
        )
    return replace(manifest, **changes)


def _with_simulation(
    manifest: SessionManifest, simulation: SimulationConfig, seed: int | None
) -> SessionManifest:
    """Attach the shared simulation settings; a file's own seed wins over them."""
    if seed is not None:
        simulation = replace(simulation, seed=seed)
    return replace(manifest, simulation=simulation)


def _read_mapping(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return data


__all__ = [
    "BARE_ROBOT_CAMERA_SIZE",
    "BARE_ROBOT_MAX_STEPS",
    "manifest_for_robot",
    "manifest_from_task_file",
    "manifest_from_world_file",
    "with_overrides",
]
