"""The session manifest: one YAML file binding robots, scene, task, policy,
backend, and viewer options for a run.

A single-robot run is just a manifest with one entry in ``robots``; a
``world`` block turns a session into one shared MuJoCo world with N robots.
`physai.runtime.create_session` builds a session from a manifest, and
`physai.config.compat` converts the older `configs/tasks/<robot>/*.yaml` and
`configs/worlds/*.yaml` files into one (see `physai.config.legacy` for their
loaders).

Schema (YAML)::

    schema_version: 1                # required; unknown versions fail loudly

    simulation:                      # optional, defaults as SimulationConfig
      seed: 0
      domain_randomization:
        enabled: false

    scene:
      name: pick_place_minimal       # optional; must be a registered scene
      overrides: {}                  # optional kwargs to the scene factory

    backend: direct                  # direct | ros2_sim | ros2_real

    world:                           # optional; present = one shared world
      timestep: 0.002
      control_hz: 30.0
      add_floor: true

    robots:                          # required, non-empty
      - id: arm_1                    # required, unique
        robot: so101                 # required; must be a registered robot
        model: assets/so101/so101_new_calib_camera.xml  # required with world
        config: {}                   # optional robot env fields (max_steps, ...)
        pose:
          position: [0.0, 0.0, 0.0]
          quaternion: [1.0, 0.0, 0.0, 0.0]
        task: pick_place             # optional per-instance override
        task_kwargs: {}
        policy: scripted             # optional; default is "idle" if omitted
        policy_kwargs: {}

    task: pick_place                 # optional global default task
    task_kwargs: {}
    success_hold_steps: 10           # optional; steps a success must hold
    policy: idle                     # optional global default policy

    viewer:
      mode: none                     # none | native | web | both
      host: 127.0.0.1
      port: 8000

Validation performed at load time (see `load_manifest`):

1. ``schema_version`` must be present and equal to a version this build
   understands.
2. ``robots`` must be non-empty; every ``id`` must be unique.
3. Every ``robot`` name must resolve through the robot registry.
4. Every resolved ``task`` name (per-instance or global) must resolve
   through the task registry.
5. Every resolved ``policy`` name (per-instance or global, unless "idle")
   must resolve through the policy registry.
6. If ``scene.name`` is given, it must be compatible with every instance's
   registered embodiment kind and resolved task (checked without
   constructing any robot, via the scene and robot registries).
7. ``backend`` must be one of ``direct``, ``ros2_sim``, ``ros2_real``.
   ``ros2_real`` is accepted by the schema for forward compatibility but
   raises a clear "not yet implemented" error (see ROADMAP.md: simulation
   only for now).
8. ``viewer.mode`` must be one of ``none``, ``native``, ``web``, ``both``.
9. Several robots share one world (default ``world`` settings unless the
   block sets them), and every robot in a world names its ``model``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..policy.registry import available_policies
from ..robots.registry import available_robots, robot_kind
from ..sim.scenes.common import REPO_ROOT
from ..sim.scenes.registry import get_scene_definition
from ..tasks.registry import available_tasks
from .legacy import SimulationConfig, _parse_simulation_config

SCHEMA_VERSION = 1
_BACKENDS = ("direct", "ros2_sim", "ros2_real")
_VIEWER_MODES = ("none", "native", "web", "both")


@dataclass(frozen=True)
class SessionRobotPose:
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class SessionRobotConfig:
    """One robot instance in a session: id, embodiment, pose, and its
    optional per-instance task/policy overrides."""

    id: str
    robot: str
    model: Path | None = None
    config: dict[str, Any] = field(default_factory=dict)
    pose: SessionRobotPose = field(default_factory=SessionRobotPose)
    task: str | None = None
    task_kwargs: dict[str, Any] = field(default_factory=dict)
    policy: str | None = None
    policy_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionSceneConfig:
    name: str | None = None
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionWorldConfig:
    """Settings of a shared MuJoCo world: one model, one clock, N robots."""

    timestep: float = 0.002
    control_hz: float = 30.0
    add_floor: bool = True


@dataclass(frozen=True)
class SessionViewerConfig:
    mode: str = "none"
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass(frozen=True)
class SessionManifest:
    """A complete, validated session: robot instances, scene, task, policy,
    backend, and viewer options."""

    schema_version: int
    robots: tuple[SessionRobotConfig, ...]
    backend: str = "direct"
    scene: SessionSceneConfig = field(default_factory=SessionSceneConfig)
    task: str | None = None
    task_kwargs: dict[str, Any] = field(default_factory=dict)
    success_hold_steps: int | None = None
    world: SessionWorldConfig | None = None
    policy: str = "idle"
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    viewer: SessionViewerConfig = field(default_factory=SessionViewerConfig)

    def task_for(self, instance: SessionRobotConfig) -> str | None:
        """The effective task for one instance: its own, or the session default."""
        return instance.task if instance.task is not None else self.task

    def policy_for(self, instance: SessionRobotConfig) -> str:
        """The effective policy for one instance: its own, or the session default."""
        return instance.policy if instance.policy is not None else self.policy


def load_manifest(path: str | Path) -> SessionManifest:
    """Load and validate a session manifest from a YAML mapping."""
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return parse_manifest(data, config_path)


def parse_manifest(data: Any, config_path: Path) -> SessionManifest:
    """Validate manifest data; ``config_path`` names the source in errors."""
    if not isinstance(data, dict):
        raise ValueError(f"manifest root must be a mapping: {config_path}")

    schema_version = data.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"{config_path}: unsupported schema_version {schema_version!r}; "
            f"this build understands {SCHEMA_VERSION}"
        )

    simulation = _parse_simulation_config(data.get("simulation", {}), config_path)
    backend = data.get("backend", "direct")
    if backend not in _BACKENDS:
        raise ValueError(
            f"{config_path}: unknown backend {backend!r}; available: "
            f"{', '.join(_BACKENDS)}"
        )
    if backend == "ros2_real":
        raise ValueError(
            "backend='ros2_real' is not yet implemented; simulation only for "
            "now (see ROADMAP.md)"
        )

    scene_data = data.get("scene") or {}
    if not isinstance(scene_data, dict):
        raise ValueError(f"{config_path}: manifest field 'scene' must be a mapping")
    overrides = _tupled(scene_data.get("overrides", {}), "scene.overrides", config_path)
    if "robot_xml" in overrides:
        overrides["robot_xml"] = _resolve_path(overrides["robot_xml"])
    scene = SessionSceneConfig(name=scene_data.get("name"), overrides=overrides)

    raw_robots = data.get("robots")
    if not isinstance(raw_robots, list) or not raw_robots:
        raise ValueError(
            f"{config_path}: manifest field 'robots' must be a non-empty list"
        )

    known_robots = available_robots()
    seen_ids: set[str] = set()
    robots: list[SessionRobotConfig] = []
    for index, raw in enumerate(raw_robots):
        if not isinstance(raw, dict):
            raise ValueError(f"{config_path}: robots[{index}] must be a mapping")
        instance_id = _required_string(raw, "id", config_path)
        if instance_id in seen_ids:
            raise ValueError(f"{config_path}: duplicate robot id {instance_id!r}")
        seen_ids.add(instance_id)

        robot_name = _required_string(raw, "robot", config_path)
        if robot_name not in known_robots:
            raise ValueError(
                f"{config_path}: unknown robot {robot_name!r}; available: "
                f"{', '.join(known_robots)}"
            )

        pose_data = raw.get("pose") or {}
        pose = SessionRobotPose(
            position=_vector(
                pose_data.get("position", (0.0, 0.0, 0.0)), 3, config_path
            ),
            quaternion=_vector(
                pose_data.get("quaternion", (1.0, 0.0, 0.0, 0.0)), 4, config_path
            ),
        )
        model = raw.get("model")
        robots.append(
            SessionRobotConfig(
                id=instance_id,
                robot=robot_name,
                model=None if model is None else _resolve_path(model),
                config=_tupled(raw.get("config", {}), "robots[].config", config_path),
                pose=pose,
                task=raw.get("task"),
                task_kwargs=dict(raw.get("task_kwargs", {})),
                policy=raw.get("policy"),
                policy_kwargs=dict(raw.get("policy_kwargs", {})),
            )
        )

    task = data.get("task")
    policy = data.get("policy", "idle")
    success_hold_steps = data.get("success_hold_steps")
    if success_hold_steps is not None and (
        not isinstance(success_hold_steps, int)
        or isinstance(success_hold_steps, bool)
        or success_hold_steps < 1
    ):
        raise ValueError(
            f"{config_path}: success_hold_steps must be a positive integer"
        )
    world = _parse_world(data.get("world"), config_path)
    if world is None and len(robots) > 1:
        world = SessionWorldConfig()  # several robots always share one world
    if world is not None:
        missing = [robot.id for robot in robots if robot.model is None]
        if missing:
            raise ValueError(
                f"{config_path}: world robots need a 'model': {', '.join(missing)}"
            )

    viewer_data = data.get("viewer") or {}
    if not isinstance(viewer_data, dict):
        raise ValueError(f"{config_path}: manifest field 'viewer' must be a mapping")
    viewer_mode = viewer_data.get("mode", "none")
    if viewer_mode not in _VIEWER_MODES:
        raise ValueError(
            f"{config_path}: unknown viewer.mode {viewer_mode!r}; available: "
            f"{', '.join(_VIEWER_MODES)}"
        )
    viewer = SessionViewerConfig(
        mode=viewer_mode,
        host=viewer_data.get("host", "127.0.0.1"),
        port=int(viewer_data.get("port", 8000)),
    )

    manifest = SessionManifest(
        schema_version=schema_version,
        robots=tuple(robots),
        backend=backend,
        scene=scene,
        task=task,
        task_kwargs=dict(data.get("task_kwargs", {})),
        success_hold_steps=success_hold_steps,
        world=world,
        policy=policy,
        simulation=simulation,
        viewer=viewer,
    )
    _validate_names_and_compatibility(manifest, config_path)
    return manifest


def _validate_names_and_compatibility(manifest: SessionManifest, source: Path) -> None:
    known_tasks = available_tasks()
    known_policies = available_policies()
    scene_definition = (
        get_scene_definition(manifest.scene.name) if manifest.scene.name else None
    )
    for instance in manifest.robots:
        task_name = manifest.task_for(instance)
        if task_name is not None and task_name not in known_tasks:
            raise ValueError(
                f"{source}: robot {instance.id!r} has unknown task {task_name!r}; "
                f"available: {', '.join(known_tasks)}"
            )
        policy_name = manifest.policy_for(instance)
        if policy_name not in known_policies and policy_name != "idle":
            raise ValueError(
                f"{source}: robot {instance.id!r} has unknown policy {policy_name!r}; "
                f"available: {', '.join((*known_policies, 'idle'))}"
            )
        if scene_definition is not None:
            kind = robot_kind(instance.robot)
            if not scene_definition.supports(kind, task_name):
                raise ValueError(
                    f"{source}: scene {manifest.scene.name!r} is incompatible with "
                    f"robot {instance.id!r} (kind={kind!r}, task={task_name!r})"
                )


def _parse_world(data: Any, source: Path) -> SessionWorldConfig | None:
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError(f"{source}: manifest field 'world' must be a mapping")
    world = SessionWorldConfig(
        timestep=float(data.get("timestep", 0.002)),
        control_hz=float(data.get("control_hz", 30.0)),
        add_floor=data.get("add_floor", True),
    )
    if world.timestep <= 0 or world.control_hz <= 0:
        raise ValueError(f"{source}: world timestep and control_hz must be positive")
    if not isinstance(world.add_floor, bool):
        raise ValueError(f"{source}: world add_floor must be a boolean")
    return world


def _tupled(value: Any, name: str, source: Path) -> dict[str, Any]:
    """A copy of a mapping with list values as tuples.

    Typed configs declare fixed-size fields as tuples, while YAML can only
    write lists, so the conversion happens once here instead of per field.
    """
    if not isinstance(value, dict):
        raise ValueError(f"{source}: {name} must be a mapping")
    return {
        key: tuple(item) if isinstance(item, list) else item
        for key, item in value.items()
    }


def _resolve_path(value: Any) -> Path:
    """A model path: absolute, else relative to the working directory or the repo."""
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError("model and robot_xml paths must be non-empty")
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    for base in (Path.cwd(), REPO_ROOT):
        if (base / candidate).exists():
            return (base / candidate).resolve()
    return (REPO_ROOT / candidate).resolve()


def _required_string(data: dict[str, Any], key: str, source: Path) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source}: manifest field {key!r} must be a non-empty string")
    return value


def _vector(value: Any, size: int, source: Path) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{source}: pose field must contain {size} values")
    return tuple(float(item) for item in value)


__all__ = [
    "SCHEMA_VERSION",
    "SessionManifest",
    "SessionRobotConfig",
    "SessionRobotPose",
    "SessionSceneConfig",
    "SessionViewerConfig",
    "SessionWorldConfig",
    "load_manifest",
    "parse_manifest",
]
