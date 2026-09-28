"""SO-101 robot environment backed by Isaac Sim (`physai.sim.isaac`).

A robot-owned backend, like `SO101Env`: observation, action, and lifecycle
only, no task (`TaskRuntime` wraps either backend the same way). Only this
module and `physai.sim.isaac` may import `isaacsim`/`omni`/`pxr`
(`pyproject.toml`'s import-linter contracts enforce this), so nothing above
the `RobotPort` boundary needs Isaac installed to import `physai.robots`.

Deliberately smaller than `SO101Env`: no Cartesian jog, task/scene objects
beyond an optional grasp cube (`cfg.cube`, ROADMAP.md's 2E tier-3 parity
test only — not a `ManipulationSceneConfig` port; no table, target, layout,
or randomization). `self.kin` is a real `ArmKinematics`, backed by a MuJoCo
model loaded purely as an FK/IK math tool and never simulated — Isaac's own
physics runs everything; this only reuses already-verified kinematics math
instead of reimplementing it against USD/PhysX (importing `mujoco` here is
fine: only `isaacsim`/`omni`/`pxr` are import-linter-restricted to this
module and `physai.sim.isaac`, and `mujoco` is a base dependency regardless).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
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
from ...sim.isaac.description import (
    apply_actuators,
    apply_cameras,
    apply_contact_friction,
    apply_frames,
)
from ...sim.isaac.description import import_robot as isaac_import_robot
from ...sim.isaac.objects import add_cube, prim_world_position
from ...sim.isaac.scene import add_ground_plane, add_studio_lighting
from ..base import RobotSpec, RobotTrainingContract
from ..description import RobotDescription, load_robot_description
from .contracts import (
    ALL_JOINT_NAMES,
    ARM_JOINT_NAMES,
    GRIPPER_JOINT_NAME,
    so101_training_contract,
)
from .kinematics import ArmKinematics

REPO_ROOT = Path(__file__).resolve().parents[4]
_DESCRIPTION_PATH = Path(__file__).resolve().parent / "description.yaml"
HOME_QPOS = np.array([0.0, -1.05, 1.25, 0.75, 0.0], dtype=np.float64)


@dataclass
class GraspCubeConfig:
    """A minimal graspable cube for the tier-3 grasp-hold parity test
    (`ROADMAP.md`'s 2E) — not a `ManipulationSceneConfig` port: no table,
    target, layout, or randomization, just enough to grasp-and-hold. `position`
    and `friction` default to `PickPlaceMinimalSceneConfig`'s own values
    (`cube_pos`, the sliding-friction component of `add_cube`'s
    `friction=[1.2, ...]`), placed directly on Isaac's ground plane rather
    than on a modeled table.
    """

    position: tuple[float, float, float] = (0.20, 0.08, 0.014)
    half_size: float = 0.014
    mass: float = 0.03
    friction: float = 1.2


@dataclass
class IsaacEnvConfig:
    """SO-101-on-Isaac simulation and observation settings.

    Deliberately narrower than `so101.mujoco_env.EnvConfig`: no `scene` (no
    task objects beyond the optional `cube` below) and no domain
    randomization (Isaac's own randomization tooling is a separate
    integration).
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
    cube: GraspCubeConfig | None = None


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
        apply_contact_friction(self.stage, self.description)
        self._camera_prims = apply_cameras(self.stage, self.description)

        self._cube_path: str | None = None
        if self.cfg.cube is not None:
            self._cube_path = add_cube(
                self.stage,
                "/World_cube",
                position=self.cfg.cube.position,
                half_size=self.cfg.cube.half_size,
                mass=self.cfg.cube.mass,
                friction=self.cfg.cube.friction,
            )

        self.kin = self._build_kinematics_oracle()

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

    def _build_kinematics_oracle(self) -> ArmKinematics:
        """A MuJoCo model used purely as an FK/IK math tool, never simulated.

        Loads the same MJCF the MuJoCo backend simulates (so joint and site
        names match), only to reuse `ArmKinematics`'s existing FK/IK math
        instead of reimplementing it against USD/PhysX. Isaac's own physics
        (`self.articulation`, `self.core`) is what actually runs the robot.
        """
        mjcf_path = self.cfg.assets_root / self.description.mjcf
        model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        return ArmKinematics(model)

    @property
    def cube_pos(self) -> np.ndarray:
        if self._cube_path is None:
            raise AttributeError("this SO101IsaacEnv has no cube (cfg.cube is None)")
        return prim_world_position(self.stage, self._cube_path)

    def _arm_qpos(self) -> np.ndarray:
        return np.asarray(self.articulation.get_dof_positions())[0, self._arm_indices]

    @property
    def ee_pos(self) -> np.ndarray:
        pos, _ = self.kin.qpos_to_site_pose(self._arm_qpos())
        return pos

    def pinch_center(self) -> np.ndarray:
        """Where the gripper's pinch point currently is, from observed joint state."""
        return self.kin.pinch_center_from_qpos(self._arm_qpos())

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
            capabilities=("joint_position", "arm_kinematics", "gripper")
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


__all__ = ["GraspCubeConfig", "IsaacEnvConfig", "SO101IsaacEnv"]
