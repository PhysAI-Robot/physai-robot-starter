"""SO-101 robot environment backed by Isaac Sim (`physai.sim.isaac`).

A robot-owned backend, like `SO101Env`: observation, action, and lifecycle
only, no task (`TaskRuntime` wraps either backend the same way). Only this
module and `physai.sim.isaac` may import `isaacsim`/`omni`/`pxr`
(`pyproject.toml`'s import-linter contracts enforce this), so nothing above
the `RobotPort` boundary needs Isaac installed to import `physai.robots`.

Kept deliberately smaller than `SO101Env` for now: no Cartesian jog, IK, or
privileged world state (cube/target positions) — `ArmKinematics` is a MuJoCo
model today (see the plan's Phase 1.4 note on why that signature change is
deferred), and a scene with graspable objects is Phase 3 work. This env
proves out the `RobotPort` contract (joint-position actions, joint-state and
image observations) against real Isaac Sim control.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ...contracts import (
    Action,
    GripperCommand,
    Header,
    ImageFrame,
    JointState,
    Observation,
)
from ...sim.isaac.core import IsaacSimulationCore, ensure_simulation_app
from ...sim.isaac.description import apply_actuators, apply_cameras, apply_frames
from ...sim.isaac.description import import_robot as isaac_import_robot
from ...sim.isaac.scene import add_ground_plane, add_studio_lighting
from ..base import RobotSpec, RobotTrainingContract
from ..description import RobotDescription, load_robot_description
from .contracts import (
    ALL_JOINT_NAMES,
    ARM_JOINT_NAMES,
    GRIPPER_JOINT_NAME,
    so101_training_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
_DESCRIPTION_PATH = Path(__file__).resolve().parent / "description.yaml"
HOME_QPOS = np.array([0.0, -1.05, 1.25, 0.75, 0.0], dtype=np.float64)


@dataclass
class IsaacEnvConfig:
    """SO-101-on-Isaac simulation and observation settings.

    Deliberately narrower than `so101.mujoco_env.EnvConfig`: no `scene` (no task
    objects yet, see this module's docstring) and no domain randomization
    (Isaac's own randomization tooling is a separate integration).
    """

    description: RobotDescription | None = None
    assets_root: Path = field(default_factory=lambda: REPO_ROOT / "assets" / "so101")
    usd_out_dir: Path = field(
        default_factory=lambda: REPO_ROOT / ".isaac_cache" / "so101"
    )
    physics_variant: str = "physx"
    control_hz: float = 30.0
    physics_hz: float = 60.0
    render: bool = True
    headless: bool = True
    cameras: tuple[str, ...] = ("wrist",)
    max_steps: int = 400
    seed: int | None = None
    gripper_force_limit: float = 0.3


class SO101IsaacEnv:
    """SO-101 embodiment environment backed by Isaac Sim's PhysX/Newton solver."""

    def __init__(self, cfg: IsaacEnvConfig | None = None) -> None:
        self.cfg = cfg or IsaacEnvConfig()
        self.description = self.cfg.description or load_robot_description(
            _DESCRIPTION_PATH
        )

        ensure_simulation_app(headless=self.cfg.headless)
        # Imported lazily, after ensure_simulation_app(): isaacsim's own
        # submodules are only import-safe once SimulationApp has started.
        from isaacsim.core.experimental.prims import Articulation

        usd_path, self.robot_prim_path = isaac_import_robot(
            self.description,
            self.cfg.assets_root,
            usd_out_dir=self.cfg.usd_out_dir,
            variant=self.cfg.physics_variant,
        )
        self.usd_path = usd_path

        import omni.usd
        from pxr import UsdPhysics

        self.stage = omni.usd.get_context().get_stage()
        if not any(p.IsA(UsdPhysics.Scene) for p in self.stage.Traverse()):
            UsdPhysics.Scene.Define(self.stage, "/physicsScene")

        add_studio_lighting(self.stage)
        add_ground_plane(self.stage)
        apply_frames(self.stage, self.description)
        self._camera_prims = apply_cameras(self.stage, self.description)

        first_camera = self.description.cameras[0] if self.description.cameras else None
        self.core = IsaacSimulationCore(
            control_hz=self.cfg.control_hz,
            physics_hz=self.cfg.physics_hz,
            render=self.cfg.render,
            camera_width=first_camera.width if first_camera else 320,
            camera_height=first_camera.height if first_camera else 240,
        )
        self.core.reset_simulation()

        self.articulation = Articulation(self.robot_prim_path)
        apply_actuators(self.articulation, self.description)
        self.dof_names = list(self.articulation.dof_names)
        self._arm_indices = [self.dof_names.index(name) for name in ARM_JOINT_NAMES]
        self._gripper_index = self.dof_names.index(GRIPPER_JOINT_NAME)
        lower, upper = self.articulation.get_dof_limits()
        self.arm_limits = np.stack(
            [
                np.asarray(lower)[0, self._arm_indices],
                np.asarray(upper)[0, self._arm_indices],
            ],
            axis=1,
        )
        self.grip_limits = (
            float(np.asarray(lower)[0, self._gripper_index]),
            float(np.asarray(upper)[0, self._gripper_index]),
        )
        if self.cfg.gripper_force_limit is not None:
            max_effort = np.asarray(self.articulation.get_dof_max_efforts()).copy()
            max_effort[0, self._gripper_index] = self.cfg.gripper_force_limit
            self.articulation.set_dof_max_efforts(max_effort)

        self.rng = np.random.default_rng(self.cfg.seed)
        self.step_count = 0
        self._last_action = Action(joint_position=HOME_QPOS.copy())

    @property
    def robot_spec(self) -> RobotSpec:
        return RobotSpec(
            name="so101",
            kind="fixed_base_manipulator",
            joint_names=ALL_JOINT_NAMES,
            action_joint_names=ARM_JOINT_NAMES,
            action_modes=("joint_position",),
            observation_modalities=("state", "images")
            if self.cfg.cameras
            else ("state",),
            capabilities=("joint_position", "gripper")
            + (("images",) if self.cfg.cameras else ()),
            joint_limits={
                name: (float(lo), float(hi))
                for name, (lo, hi) in zip(ARM_JOINT_NAMES, self.arm_limits)
            },
            max_joint_delta={name: 0.75 for name in ARM_JOINT_NAMES},
            metadata={"control_hz": self.cfg.control_hz, "simulator": "isaac"},
            joint_state_frame="base",
            camera_frames={name: f"camera_{name}" for name in self.cfg.cameras},
            units={
                "joint_position": "rad",
                "joint_velocity": "rad/s",
                "position": "m",
            },
        )

    @property
    def training_contract(self) -> RobotTrainingContract:
        camera_config = {
            name: {
                "width": self.core.camera_size[0],
                "height": self.core.camera_size[1],
            }
            for name in self.cfg.cameras
        }
        return so101_training_contract(camera_config=camera_config)

    def gripper_to_joint(self, normalized: float) -> float:
        lo, hi = self.grip_limits
        return float(lo + np.clip(normalized, 0.0, 1.0) * (hi - lo))

    def joint_to_gripper(self, q: float) -> float:
        lo, hi = self.grip_limits
        return float(np.clip((q - lo) / (hi - lo), 0.0, 1.0))

    def reset(self, seed: int | None = None) -> Observation:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.core.reset_simulation()
        target = np.zeros((1, len(self.dof_names)), dtype=np.float32)
        target[0, self._arm_indices] = HOME_QPOS
        target[0, self._gripper_index] = self.gripper_to_joint(1.0)
        self.articulation.set_dof_position_targets(target)
        self.articulation.set_dof_positions(target)
        self._last_action = Action(
            joint_position=HOME_QPOS.copy(), gripper=GripperCommand(position=1.0)
        )
        return self.observe()

    def _joint_state(self) -> JointState:
        positions = np.asarray(self.articulation.get_dof_positions())[0]
        velocities = np.asarray(self.articulation.get_dof_velocities())[0]
        efforts = np.asarray(self.articulation.get_dof_efforts())[0]
        order = self._arm_indices + [self._gripper_index]
        return JointState(
            name=ALL_JOINT_NAMES,
            position=positions[order],
            velocity=velocities[order],
            effort=efforts[order],
            header=Header(
                stamp=self.step_count * self.core.control_dt, frame_id="base"
            ),
        )

    def render_camera(self, name: str) -> np.ndarray:
        prim_path = self._camera_prims.get(name)
        if prim_path is None:
            raise KeyError(f"camera {name!r} not attached to this description")
        return self.core.render_camera(prim_path)

    def observe(self) -> Observation:
        images: dict[str, ImageFrame] = {}
        if self.core.render_enabled:
            for camera in self.cfg.cameras:
                if camera not in self._camera_prims:
                    continue
                images[camera] = ImageFrame(
                    data=self.render_camera(camera),
                    camera_name=camera,
                    header=Header(
                        stamp=self.step_count * self.core.control_dt,
                        frame_id=f"camera_{camera}",
                    ),
                )
        return Observation(
            joint_state=self._joint_state(),
            images=images,
            step=self.step_count,
            sim_time=self.step_count * self.core.control_dt,
        )

    def send_action(self, action: Action) -> None:
        if action.joint_position is None:
            raise ValueError("SO101IsaacEnv requires joint-position actions")
        target = np.asarray(self.articulation.get_dof_position_targets()).copy()
        target[0, self._arm_indices] = action.joint_position
        gripper = action.gripper or GripperCommand()
        target[0, self._gripper_index] = self.gripper_to_joint(gripper.clipped())
        self.articulation.set_dof_position_targets(target)
        self._last_action = action

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict]:
        self.send_action(action)
        self.core.step_simulation()
        self.step_count = self.core.step_count
        observation = self.observe()
        truncated = self.step_count >= self.cfg.max_steps
        return observation, 0.0, False, truncated, {}

    def close(self) -> None:
        self.core.close()


__all__ = ["IsaacEnvConfig", "SO101IsaacEnv"]
