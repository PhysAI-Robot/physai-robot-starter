"""ViperX 300 S (ALOHA follower arm) environment backed by Isaac Sim.

The arm is not imported from a URDF: there is none. `sim.isaac.description.import_robot`
converts the Menagerie MJCF with Isaac's own MJCF importer (see that module's notes on
what the importer does and does not carry over), and everything above it is the
SO-101's Isaac environment. This class reuses `SO101IsaacEnv`'s scene, camera, cube,
reset and stepping machinery and overrides only what differs for this robot: six arm
joints, a prismatic finger pair measured in metres, gravity compensation, and a
robot spec of its own.

Only `physai.sim.isaac` and the two `isaac_env` modules import `isaacsim`/`omni`/`pxr`
(the import-linter contract in `pyproject.toml`); this module reaches `pxr` and `omni`
only through `sim.isaac`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import mujoco
import numpy as np

from ...contracts import (
    Action,
    GripperCommand,
    Header,
    JointState,
    Observation,
    parse_camera_resolution,
)
from ...sim.isaac.core import IsaacSimulationCore, ensure_simulation_app
from ...sim.isaac.description import (
    apply_actuators,
    apply_cameras,
    apply_contact_friction,
    apply_contact_pad_colliders,
    apply_frames,
    apply_gravity_compensation,
    articulation_root_path,
    ensure_physics_scene,
    get_stage,
)
from ...sim.isaac.description import import_robot as isaac_import_robot
from ...sim.isaac.scene import (
    add_ground_plane,
    add_studio_lighting,
    add_workspace,
    checker_texture,
)
from ..base import RobotSpec, RobotTrainingContract
from ..description import load_robot_description
from ..so101.isaac_env import IsaacEnvConfig, SO101IsaacEnv
from .contracts import (
    ALL_JOINT_NAMES,
    ARM_JOINT_NAMES,
    GRIPPER_JOINT_NAME,
    viperx300s_training_contract,
)
from .kinematics import ViperX300SKinematics
from .mujoco_env import (
    DEFAULT_CUBE_X_RANGE,
    DEFAULT_CUBE_Y_RANGE,
    DEFAULT_TARGET_X_RANGE,
    DEFAULT_TARGET_Y_RANGE,
    HOME_QPOS,
)
from .scene import scene_defaults

REPO_ROOT = Path(__file__).resolve().parents[4]
_DESCRIPTION_PATH = Path(__file__).resolve().parent / "description.yaml"
_LEFT_FINGER = "left_finger"
_RIGHT_FINGER = "right_finger"


@dataclass
class ViperX300SIsaacEnvConfig(IsaacEnvConfig):
    """ViperX 300 S-on-Isaac settings: the SO-101 Isaac config with this robot's defaults."""

    assets_root: Path = field(
        default_factory=lambda: REPO_ROOT / "assets" / "viperx300s"
    )
    usd_out_dir: Path = field(
        default_factory=lambda: REPO_ROOT / ".isaac_cache" / "viperx300s"
    )
    cameras: tuple[str, ...] = ("front", "wrist")
    max_steps: int = 600
    # None keeps the MJCF's own +-35 N finger force limit (the SO-101's 0.3 is its units).
    gripper_force_limit: float | None = None
    cube_x_range: tuple[float, float] = DEFAULT_CUBE_X_RANGE
    cube_y_range: tuple[float, float] = DEFAULT_CUBE_Y_RANGE
    target_x_range: tuple[float, float] = DEFAULT_TARGET_X_RANGE
    target_y_range: tuple[float, float] = DEFAULT_TARGET_Y_RANGE
    # The MuJoCo env's default, for the same reason: the soft position drives sag.
    gravity_compensation: bool = True


class ViperX300SIsaacEnv(SO101IsaacEnv):
    """ViperX 300 S embodiment environment backed by Isaac Sim's PhysX."""

    def __init__(self, cfg: ViperX300SIsaacEnvConfig | None = None) -> None:
        self.cfg = cfg or ViperX300SIsaacEnvConfig()
        scene = self.cfg.scene
        if scene is not None:
            defaults = scene_defaults()
            missing = {
                key: value
                for key, value in defaults.items()
                if getattr(scene, key) is None
            }
            if missing:
                self.cfg = replace(self.cfg, scene=replace(scene, **missing))
            scene = self.cfg.scene
            if (
                getattr(scene, "floor_style", "checker") != "checker"
                or any(getattr(scene, "floor_offset", (0.0, 0.0)))
                or getattr(scene, "floor_yaw_deg", 0.0)
            ):
                raise ValueError(
                    "Isaac Sim only draws the unmoved checker floor "
                    "(floor_style, floor_offset, floor_yaw_deg)"
                )
            cube_names = [cube.name for cube in scene.cubes()]
            if len(cube_names) > 1:
                raise ValueError(
                    "ViperX300SIsaacEnv supports single-cube scenes only; "
                    f"this scene has cubes {tuple(cube_names)}"
                )
        if self.cfg.randomize_cube is None:
            self.cfg = replace(self.cfg, randomize_cube=self.cfg.scene is not None)
        self.description = self.cfg.description or load_robot_description(
            _DESCRIPTION_PATH
        )

        ensure_simulation_app(headless=self.cfg.headless)
        from isaacsim.core.experimental.prims import Articulation

        usd_path, self.robot_prim_path = isaac_import_robot(
            self.description,
            self.cfg.assets_root,
            usd_out_dir=self.cfg.usd_out_dir,
            variant=self.cfg.physics_variant,
        )
        self.usd_path = usd_path
        self.stage = get_stage()
        ensure_physics_scene(self.stage)
        if self.cfg.gravity_compensation:
            apply_gravity_compensation(self.stage, self.robot_prim_path)

        add_studio_lighting(self.stage, scale=self.cfg.lighting_scale)
        add_ground_plane(
            self.stage,
            texture_path=checker_texture(
                self.cfg.usd_out_dir.parent / "studio" / "floor_checker.png"
            ),
        )
        apply_frames(self.stage, self.description)
        apply_contact_friction(self.stage, self.description)
        apply_contact_pad_colliders(self.stage, self.description)
        self._render_size = parse_camera_resolution(self.cfg.camera_resolution)
        self._camera_prims = apply_cameras(
            self.stage, self.description, size=self._render_size
        )

        self._cube_path: str | None = None
        self._cube_body = None
        self._target_pad_path: str | None = None
        if self.cfg.scene is not None:
            self._target_pos = np.asarray(self.cfg.scene.target_pos, dtype=np.float64)
            prims = add_workspace(
                self.stage,
                self.cfg.scene,
                width=self._render_size[0],
                height=self._render_size[1],
            )
            self._camera_prims["front"] = prims.front_camera
            self._cube_path = prims.cubes[0] if prims.cubes else None
            self._target_pad_path = prims.target_pad

        self.kin = self._build_kinematics_oracle()
        # Display/telemetry mirror for `physai.web.Host`, never simulated.
        self.model = self.kin.model
        self.data = mujoco.MjData(self.model)
        cube_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free"
        )
        self._mirror_cube_qadr = (
            int(self.model.jnt_qposadr[cube_joint]) if cube_joint >= 0 else None
        )

        self.core = IsaacSimulationCore(
            control_hz=self.cfg.control_hz,
            physics_hz=self.cfg.physics_hz,
            render=self.cfg.render,
            camera_width=self._render_size[0],
            camera_height=self._render_size[1],
        )
        self.core.reset_simulation()

        if self._cube_path is not None:
            from isaacsim.core.experimental.prims import RigidPrim

            self._cube_body = RigidPrim(self._cube_path)

        self.articulation = Articulation(
            articulation_root_path(self.stage, self.robot_prim_path)
        )
        apply_actuators(self.articulation, self.description)
        self.dof_names = list(self.articulation.dof_names)
        self._arm_indices = [self.dof_names.index(name) for name in ARM_JOINT_NAMES]
        # `left_finger` is the robot's "gripper"; `right_finger` mirrors it.
        self._gripper_index = self.dof_names.index(_LEFT_FINGER)
        self._right_index = self.dof_names.index(_RIGHT_FINGER)
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
            max_effort[0, self._right_index] = self.cfg.gripper_force_limit
            self.articulation.set_dof_max_efforts(max_effort)

        self.rng = np.random.default_rng(self.cfg.seed)
        self.step_count = 0
        self._last_action = Action(joint_position=HOME_QPOS.copy())
        if self.core.render_enabled and self._camera_prims:
            self._require_rendered_robot()

    def _build_kinematics_oracle(self) -> ViperX300SKinematics:
        """A MuJoCo model used only as an FK/IK tool (see the SO-101 env's note)."""
        if self.cfg.scene is not None:
            model, _ = self.cfg.scene.build_model()
        else:
            model = mujoco.MjModel.from_xml_path(
                str(self.cfg.assets_root / self.description.mjcf)
            )
        return ViperX300SKinematics(model, ee_site=self.description.ee_site)

    def _render_check_camera(self) -> str:
        # The wrist camera looks away from the arm; the front camera sees all of it.
        return (
            "front" if "front" in self._camera_prims else super()._render_check_camera()
        )

    @property
    def robot_spec(self) -> RobotSpec:
        return RobotSpec(
            name="viperx300s",
            kind="fixed_base_manipulator",
            joint_names=ALL_JOINT_NAMES,
            action_joint_names=ARM_JOINT_NAMES,
            action_modes=("joint_position",),
            observation_modalities=("state", "images", "ee_pose")
            if self.cfg.cameras
            else ("state", "ee_pose"),
            capabilities=("joint_position", "arm_kinematics", "gripper")
            + (("images",) if self.cfg.cameras else ()),
            joint_limits={
                name: (float(lo), float(hi))
                for name, (lo, hi) in zip(ARM_JOINT_NAMES, self.kin.limits)
            },
            max_joint_delta={name: 0.75 for name in ARM_JOINT_NAMES},
            metadata={
                "control_hz": self.cfg.control_hz,
                "action_schema": "viperx300s.joint_position.v1",
                "simulator": "isaac",
                "gripper_unit": "m",
            },
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
                "encoding": "rgb8",
            }
            for name in self.cfg.cameras
        }
        return viperx300s_training_contract(camera_config=camera_config)

    def _set_gripper_target(self, target: np.ndarray, finger_m: float) -> None:
        target[0, self._gripper_index] = finger_m
        target[0, self._right_index] = -finger_m

    def reset(self, seed: int | None = None) -> Observation:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.core.reset_simulation()
        cube_xy = self._reset_cube()
        if self.cfg.randomize_target:
            self._reset_target(cube_xy)
        self._shift_cameras()
        target = np.zeros((1, len(self.dof_names)), dtype=np.float32)
        target[0, self._arm_indices] = HOME_QPOS
        self._set_gripper_target(target, self.gripper_to_joint(1.0))
        self.articulation.set_dof_position_targets(target)
        self.articulation.set_dof_positions(target)
        self._last_action = Action(
            joint_position=HOME_QPOS.copy(), gripper=GripperCommand(position=1.0)
        )
        # One control step at HOME so the first images are fresh (see the SO-101 env).
        self.core.step_simulation()
        self.core.step_count = 0
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

    def _sync_mirror(self, joint_state: JointState) -> None:
        """Copy Isaac's joint positions into the MuJoCo display mirror."""
        by_name = dict(zip(joint_state.name, joint_state.position))
        finger = by_name.pop(GRIPPER_JOINT_NAME)
        by_name[_LEFT_FINGER] = finger
        by_name[_RIGHT_FINGER] = -finger
        for name, value in by_name.items():
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id >= 0:
                self.data.qpos[self.model.jnt_qposadr[joint_id]] = value
        if self._mirror_cube_qadr is not None and self._cube_body is not None:
            positions, orientations = self._cube_body.get_world_poses()
            start = self._mirror_cube_qadr
            self.data.qpos[start : start + 3] = np.asarray(positions)[0]
            self.data.qpos[start + 3 : start + 7] = np.asarray(orientations)[0]
        self.data.time = self.step_count * self.core.control_dt
        mujoco.mj_kinematics(self.model, self.data)

    def send_action(self, action: Action) -> None:
        if action.joint_position is None:
            raise ValueError("ViperX300SIsaacEnv requires joint-position actions")
        target = np.asarray(self.articulation.get_dof_position_targets()).copy()
        target[0, self._arm_indices] = np.clip(
            np.asarray(action.joint_position).reshape(len(ARM_JOINT_NAMES)),
            self.arm_limits[:, 0],
            self.arm_limits[:, 1],
        )
        gripper = action.gripper or GripperCommand()
        self._set_gripper_target(target, self.gripper_to_joint(gripper.clipped()))
        self.articulation.set_dof_position_targets(target)
        self._last_action = action


__all__ = [
    "ViperX300SIsaacEnv",
    "ViperX300SIsaacEnvConfig",
]
