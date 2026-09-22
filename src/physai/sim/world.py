"""Shared MuJoCo worlds containing multiple namespaced robot instances."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


@dataclass(frozen=True)
class RobotInstanceConfig:
    """Configuration for one robot model attached to a shared world."""

    instance_id: str
    model_path: Path
    robot_name: str = "custom"
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not isinstance(self.instance_id, str):
            raise TypeError("robot instance_id must be a string")
        if not self.instance_id:
            raise ValueError("robot instance_id must not be empty")
        if not self.robot_name:
            raise ValueError("robot instance robot_name must not be empty")
        if len(self.position) != 3:
            raise ValueError("robot instance position must have three values")
        if len(self.quaternion) != 4:
            raise ValueError("robot instance quaternion must have four values")
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"robot model not found: {self.model_path}")


@dataclass(frozen=True)
class RobotBinding:
    """Compiled MuJoCo IDs belonging to one shared-world robot instance."""

    instance_id: str
    prefix: str
    joint_ids: Mapping[str, int]
    actuator_ids: Mapping[str, int]
    qpos_addresses: Mapping[str, int]


class SharedWorld:
    """Own one composite model, data object, clock, and physics stepper.

    Robot-specific adapters consume the returned bindings and remain responsible
    for translating local observations and actions. This class deliberately does
    not merge heterogeneous robot action or observation spaces.

    `shared_attach`, if given, is called as ``shared_attach(robot_name, child)``
    for every instance being attached, letting a robot inject shared-world-only
    MJCF (e.g. extra cameras) without this class knowing any robot's name; pass
    `physai.robots.shared_attach` to wire in whatever robots have registered.
    """

    def __init__(
        self,
        instances: tuple[RobotInstanceConfig, ...],
        *,
        timestep: float = 0.002,
        control_hz: float = 30.0,
        add_floor: bool = True,
        shared_attach: Callable[[str, Any], None] | None = None,
    ) -> None:
        if not instances:
            raise ValueError("shared world requires at least one robot instance")
        ids = [instance.instance_id for instance in instances]
        if len(set(ids)) != len(ids):
            raise ValueError("robot instance_id values must be unique")
        if timestep <= 0 or control_hz <= 0:
            raise ValueError("timestep and control_hz must be positive")

        spec = mujoco.MjSpec()
        spec.option.timestep = timestep
        spec.add_texture(
            name="physai_shared_sky",
            type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
            rgb1=[0.96, 0.98, 1.0],
            rgb2=[0.76, 0.84, 0.92],
            width=256,
            height=256,
        )
        spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
        spec.visual.headlight.diffuse = [0.65, 0.65, 0.65]
        spec.worldbody.add_light(
            name="physai_shared_key",
            pos=[1.5, -2.0, 3.0],
            dir=[-0.35, 0.45, -1.0],
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            diffuse=[0.8, 0.8, 0.8],
        )
        spec.worldbody.add_light(
            name="physai_shared_fill",
            pos=[-1.0, 1.5, 2.0],
            dir=[0.3, -0.35, -1.0],
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            diffuse=[0.45, 0.5, 0.55],
        )
        if add_floor:
            spec.worldbody.add_geom(
                name="physai_shared_floor",
                type=mujoco.mjtGeom.mjGEOM_PLANE,
                size=[0.0, 0.0, 0.05],
            )

        prefixes: dict[str, str] = {}
        for instance in instances:
            child = mujoco.MjSpec.from_file(str(instance.model_path))
            prefix = f"{instance.instance_id}__"
            if shared_attach is not None:
                shared_attach(instance.robot_name, child)
            frame = spec.worldbody.add_frame(
                name=f"{instance.instance_id}__root",
                pos=list(instance.position),
                quat=list(instance.quaternion),
            )
            spec.attach(child, prefix=prefix, frame=frame)
            prefixes[instance.instance_id] = prefix

        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.control_hz = control_hz
        self.n_substeps = max(1, round((1.0 / control_hz) / self.model.opt.timestep))
        self.step_count = 0
        self._bindings = {
            instance.instance_id: self._make_binding(
                instance, prefixes[instance.instance_id]
            )
            for instance in instances
        }
        self.reset()

    @property
    def bindings(self) -> Mapping[str, RobotBinding]:
        return self._bindings

    def binding(self, instance_id: str) -> RobotBinding:
        try:
            return self._bindings[instance_id]
        except KeyError as exc:
            raise KeyError(f"unknown robot instance {instance_id!r}") from exc

    def _make_binding(self, instance: RobotInstanceConfig, prefix: str) -> RobotBinding:
        joint_ids: dict[str, int] = {}
        qpos_addresses: dict[str, int] = {}
        for joint_id in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if name is None or not name.startswith(prefix):
                continue
            local_name = name.removeprefix(prefix)
            joint_ids[local_name] = joint_id
            qpos_addresses[local_name] = int(self.model.jnt_qposadr[joint_id])

        actuator_ids: dict[str, int] = {}
        for actuator_id in range(self.model.nu):
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
            )
            if name is None or not name.startswith(prefix):
                continue
            actuator_ids[name.removeprefix(prefix)] = actuator_id

        if not joint_ids and not actuator_ids:
            raise ValueError(
                f"robot instance {instance.instance_id!r} attached no namespaced entities"
            )
        return RobotBinding(
            instance_id=instance.instance_id,
            prefix=prefix,
            joint_ids=joint_ids,
            actuator_ids=actuator_ids,
            qpos_addresses=qpos_addresses,
        )

    def reset(self) -> None:
        """Reset the complete shared world atomically."""
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        mujoco.mj_forward(self.model, self.data)

    def set_controls(self, instance_id: str, controls: Mapping[str, float]) -> None:
        """Write only the named instance's actuator slice."""
        binding = self.binding(instance_id)
        for name, value in controls.items():
            try:
                actuator_id = binding.actuator_ids[name]
            except KeyError as exc:
                raise KeyError(
                    f"robot instance {instance_id!r} has no actuator {name!r}"
                ) from exc
            if not np.isfinite(value):
                raise ValueError(f"control for actuator {name!r} is not finite")
            self.data.ctrl[actuator_id] = float(value)

    def joint_positions(self, instance_id: str) -> dict[str, float]:
        """Read local joint positions without exposing the composite vector."""
        binding = self.binding(instance_id)
        return {
            name: float(self.data.qpos[address])
            for name, address in binding.qpos_addresses.items()
        }

    def step(self) -> None:
        """Advance the composite world exactly once per control tick."""
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1

    def close(self) -> None:
        """Keep the lifecycle hook explicit for future renderer ownership."""


__all__ = ["RobotBinding", "RobotInstanceConfig", "SharedWorld"]
