"""Build a runnable session from a validated `SessionManifest`.

A manifest with one robot becomes a `RuntimeComposition` (robot, task, and
policy composed by `create_runtime`); a manifest with a `world` block becomes
one `SharedWorld` holding every robot. Callers such as `scripts/run_sim.py`
wrap the result in whatever front end they need (an episode loop or a
`Host`); nothing here knows a robot's name.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from ..config.manifest import SessionManifest, SessionRobotConfig
from ..robots.registry import create_env_config, robot_kind, shared_attach
from ..sim.scenes import default_scene_for
from ..sim.world import RobotInstanceConfig, SharedWorld
from .composition import RuntimeComposition, create_runtime

# Settings the `simulation` block owns; a robot's own `config` must not repeat
# them, so each default has exactly one source.
_SIMULATION_OWNED = ("seed", "domain_randomization")


@dataclass
class Session:
    """What a manifest builds: one composed robot, or one shared world."""

    manifest: SessionManifest
    runtime: RuntimeComposition | None = None
    world: SharedWorld | None = None
    instances: tuple[RobotInstanceConfig, ...] = field(default_factory=tuple)
    # True when the robot stopped rendering inline for a host-driven session,
    # so the host's camera thread must supply its frames.
    host_renders_cameras: bool = False

    @property
    def robot_name(self) -> str:
        """The registered robot name of a single-robot session."""
        return self.manifest.robots[0].robot

    def close(self) -> None:
        if self.runtime is not None:
            self.runtime.close()
        if self.world is not None:
            self.world.close()


def create_session(
    manifest: SessionManifest,
    *,
    render: bool | None = None,
    host_driven: bool = False,
    policy_kwargs: dict[str, Any] | None = None,
) -> Session:
    """Build the session a manifest describes.

    ``render`` overrides the robot's own render setting (a run that records
    video or serves cameras needs it on). ``host_driven`` marks a session a
    `Host` will step: the host's camera thread is then the only renderer, so
    robots that render inline are told to stop, and the task is not composed
    around the robot (the host scores none). ``policy_kwargs`` carries
    run-time policy inputs, such as a checkpoint path, that a manifest does
    not hold.
    """
    if manifest.backend != "direct":
        raise ValueError(
            f"backend {manifest.backend!r} is not supported by create_session; "
            "ROS2-backed runs use scripts/run_ros2_sim.py"
        )
    if manifest.world is not None:
        return _create_world_session(manifest)
    return _create_single_session(
        manifest,
        manifest.robots[0],
        render=render,
        host_driven=host_driven,
        policy_kwargs=policy_kwargs or {},
    )


def _create_single_session(
    manifest: SessionManifest,
    robot: SessionRobotConfig,
    *,
    render: bool | None,
    host_driven: bool,
    policy_kwargs: dict[str, Any],
) -> Session:
    policy_name = manifest.policy_for(robot)
    task_name = manifest.task_for(robot)
    scene_name = manifest.scene.name
    if host_driven and task_name is not None:
        # A Host steps the bare robot and scores no task, and wrapping one
        # would end episodes on task success. The task still picks the scene.
        scene_name = scene_name or default_scene_for(robot_kind(robot.robot), task_name)
        task_name = None
    fields_, host_renders_cameras = _robot_fields(
        manifest, robot, render=render, host_driven=host_driven
    )
    kwargs: dict[str, Any] = {
        "robot_kwargs": fields_,
        "scene_name": scene_name,
        "scene_kwargs": dict(manifest.scene.overrides),
        "task_name": task_name,
        "task_kwargs": {**manifest.task_kwargs, **robot.task_kwargs},
        "task_success_hold_steps": manifest.success_hold_steps,
        "policy_name": None if policy_name == "idle" else policy_name,
        **robot.policy_kwargs,
        **policy_kwargs,
    }
    return Session(
        manifest,
        runtime=create_runtime(robot.robot, **kwargs),
        host_renders_cameras=host_renders_cameras,
    )


def _create_world_session(manifest: SessionManifest) -> Session:
    unsupported = [
        robot.id
        for robot in manifest.robots
        if manifest.task_for(robot) is not None or manifest.policy_for(robot) != "idle"
    ]
    if unsupported:
        raise ValueError(
            "shared-world sessions do not run tasks or policies yet; remove "
            f"them from: {', '.join(unsupported)}"
        )
    world_config = manifest.world
    instances = tuple(
        RobotInstanceConfig(
            instance_id=robot.id,
            model_path=robot.model,
            robot_name=robot.robot,
            position=robot.pose.position,
            quaternion=robot.pose.quaternion,
        )
        for robot in manifest.robots
    )
    world = SharedWorld(
        instances,
        timestep=world_config.timestep,
        control_hz=world_config.control_hz,
        add_floor=world_config.add_floor,
        shared_attach=shared_attach,
    )
    return Session(manifest, world=world, instances=instances)


def _robot_fields(
    manifest: SessionManifest,
    robot: SessionRobotConfig,
    *,
    render: bool | None,
    host_driven: bool,
) -> tuple[dict[str, Any], bool]:
    """The robot's env-config fields, and whether a host now renders its cameras.

    Simulation defaults, the robot's own config, then run overrides; only
    fields the robot's config declares are injected, so the generic defaults
    never have to know which robot defines what. A robot that renders inline
    and can be told to stop (`camera_stride`) hands its cameras to the host.
    """
    duplicated = [key for key in _SIMULATION_OWNED if key in robot.config]
    if duplicated:
        raise ValueError(
            f"robot {robot.id!r} config sets {', '.join(duplicated)}; "
            "set it under 'simulation' instead"
        )
    accepted = {item.name for item in fields(create_env_config(robot.robot))}
    simulation = manifest.simulation
    result = {
        key: value
        for key, value in {
            "seed": simulation.seed,
            "domain_randomization": simulation.domain_randomization,
        }.items()
        if key in accepted
    }
    result.update(robot.config)
    if render is not None and "render" in accepted:
        result["render"] = render
    host_renders = host_driven and "camera_stride" in accepted
    if host_renders:
        result["camera_stride"] = 0
    return result, host_renders


__all__ = ["Session", "create_session"]
