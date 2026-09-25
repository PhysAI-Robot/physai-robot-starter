"""SO-101 robot environment backed by the shared MuJoCo simulation core."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property

import mujoco
import numpy as np

from ...contracts import (
    Action,
    GripperCommand,
    Header,
    ImageFrame,
    JointState,
    Observation,
    Twist,
)
from ...control.resolver import TwistToJointResolver
from ...robots.base import RobotSpec, RobotTrainingContract
from ...sim.core import MuJoCoSimulationCore
from ...sim.domain_randomization import (
    DomainRandomizationConfig,
    DomainRandomizationEngine,
    RandomizationMetadata,
)
from ...sim.scenes import ManipulationSceneConfig, PickPlaceMinimalSceneConfig
from .contracts import (
    ALL_JOINT_NAMES,
    ARM_JOINT_NAMES,
    GRIPPER_JOINT_NAME,
    so101_training_contract,
)
from .jog import resolve_jog
from .kinematics import ArmKinematics
from .layout import create_layout
from .scene import scene_defaults

HOME_QPOS = np.array([0.0, -1.05, 1.25, 0.75, 0.0], dtype=np.float64)


@dataclass
class EnvConfig:
    """SO-101-specific simulation and observation settings."""

    scene: ManipulationSceneConfig = field(
        default_factory=lambda: PickPlaceMinimalSceneConfig(**scene_defaults())
    )
    control_hz: float = 30.0
    render: bool = True
    cameras: tuple[str, ...] = ("front", "wrist")
    camera_stride: int = 1
    max_steps: int = 400
    randomize_cube: bool = True
    cube_x_range: tuple[float, float] = (0.20, 0.24)
    cube_y_range: tuple[float, float] = (0.05, 0.13)
    randomize_target: bool = False
    target_x_range: tuple[float, float] = (0.16, 0.26)
    target_y_range: tuple[float, float] = (-0.13, -0.04)
    seed: int | None = None
    domain_randomization: DomainRandomizationConfig = field(
        default_factory=DomainRandomizationConfig
    )
    # Gripper actuator torque cap (N*m), emulating a current-limited servo. The
    # model's own +/-3.35 N*m rating drives a position-controlled squeeze to
    # 20-50x a cube's weight and makes the contact chatter. visual_servo's
    # shallow grip needs the cap; the scripted expert's deeper squeeze does not,
    # but the shared default stays for its other callers. Measurements are in
    # research/scripted_experts/FINDINGS.md.
    gripper_force_limit: float = 0.3


class SO101Env(MuJoCoSimulationCore):
    """SO-101 embodiment environment for registered task scenes."""

    def __init__(self, cfg: EnvConfig | None = None) -> None:
        self.cfg = cfg or EnvConfig()
        defaults = scene_defaults()
        missing = {
            key: value
            for key, value in defaults.items()
            if getattr(self.cfg.scene, key) is None
        }
        if missing:
            self.cfg = replace(
                self.cfg,
                scene=replace(self.cfg.scene, **missing),
            )
        self.model, self.spec = self.cfg.scene.build_model()
        self.randomization = DomainRandomizationEngine(
            self.model, self.cfg.domain_randomization
        )
        super().__init__(
            self.model,
            control_hz=self.cfg.control_hz,
            render=self.cfg.render,
            camera_width=self.cfg.scene.camera_width,
            camera_height=self.cfg.scene.camera_height,
        )
        self.rng = np.random.default_rng(self.cfg.seed)

        def jid(name: str) -> int:
            index = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if index < 0:
                raise KeyError(f"joint {name!r} missing from model")
            return index

        def aid(name: str) -> int:
            index = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if index < 0:
                raise KeyError(f"actuator {name!r} missing from model")
            return index

        self.arm_joint_ids = np.array([jid(name) for name in ARM_JOINT_NAMES])
        self.gripper_joint_id = jid(GRIPPER_JOINT_NAME)
        self.arm_qadr = self.model.jnt_qposadr[self.arm_joint_ids]
        self.arm_vadr = self.model.jnt_dofadr[self.arm_joint_ids]
        self.grip_qadr = int(self.model.jnt_qposadr[self.gripper_joint_id])
        self.grip_vadr = int(self.model.jnt_dofadr[self.gripper_joint_id])
        self.arm_act_ids = np.array([aid(name) for name in ARM_JOINT_NAMES])
        self.grip_act_id = aid(GRIPPER_JOINT_NAME)
        self.arm_limits = self.model.jnt_range[self.arm_joint_ids].copy()
        self.grip_limits = self.model.jnt_range[self.gripper_joint_id].copy()
        if self.cfg.gripper_force_limit is not None:
            limit = float(self.cfg.gripper_force_limit)
            self.model.actuator_forcerange[self.grip_act_id] = [-limit, limit]

        self.layout = create_layout(self.model, self.cfg.scene)
        self.target_sid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "target_site"
        )
        self.kin = ArmKinematics(self.model, ee_site=self.cfg.scene.ee_site)
        self.jog_kin = ArmKinematics(
            self.model, ee_site="wristframe", joint_names=ARM_JOINT_NAMES[1:3]
        )
        self._jog_resolver = TwistToJointResolver(
            self.jog_kin,
            self.data,
            dt=1.0 / self.cfg.control_hz,
            position_only=True,
            # Damping this high enough to fully suppress wrist coupling
            # (~0.15) made the 2-joint solve direction-inaccurate instead:
            # e.g. holding pure -x from the home pose (near a singularity
            # for that direction) leaked most of the response into +z, so
            # "back up" visibly lifted the arm. `jog.py`'s MAX_TARGET_LEAD
            # now caps wrist coupling directly, so this only needs to keep
            # the Cartesian solve numerically stable, not do double duty.
            damping=0.03,
        )
        self._jog_target = HOME_QPOS.copy()
        self.randomization_metadata = RandomizationMetadata(
            enabled=False,
            seed=self.cfg.seed,
            friction_scale=1.0,
            mass_scale=1.0,
            lighting_scale=1.0,
            camera_position_offset={},
            clutter_position={},
        )
        self._last_action = Action(joint_position=HOME_QPOS.copy())

    @cached_property
    def robot_spec(self) -> RobotSpec:
        """Built once: the Host reads it on every tick."""
        return RobotSpec(
            name="so101",
            kind="fixed_base_manipulator",
            joint_names=ALL_JOINT_NAMES,
            action_joint_names=ARM_JOINT_NAMES,
            action_modes=("joint_position",),
            observation_modalities=("state", "images", "ee_pose"),
            capabilities=("joint_position", "arm_kinematics", "gripper", "images"),
            joint_limits={
                name: tuple(float(value) for value in limit)
                for name, limit in zip(ARM_JOINT_NAMES, self.arm_limits)
            },
            # JointRateLimiter caps command-to-command steps at 0.5 rad, but
            # this gate compares a command against the *measured* position,
            # which trails it by the servo's tracking error. Equal values would
            # reject a command the limiter considers exactly legal, so leave
            # headroom: the gate still catches IK teleports, not normal lag.
            max_joint_delta={name: 0.75 for name in ARM_JOINT_NAMES},
            metadata={
                "control_hz": self.cfg.control_hz,
                "action_schema": "so101.joint_position.v1",
            },
            joint_state_frame="base",
            camera_frames={"front": "camera_front", "wrist": "camera_wrist"},
            units={
                "joint_position": "rad",
                "joint_velocity": "rad/s",
                "position": "m",
            },
        )

    @property
    def training_contract(self) -> RobotTrainingContract:
        """Return the configured SO-101 contract for training adapters."""
        camera_config = {
            name: {
                "width": self.cfg.scene.camera_width,
                "height": self.cfg.scene.camera_height,
                "encoding": "rgb8",
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

    def resolve_twist_jog(
        self,
        twist: Twist,
        joint_state: JointState,
        gripper: GripperCommand | None = None,
    ) -> Action:
        return resolve_jog(
            twist,
            joint_state,
            cartesian_resolver=self._jog_resolver,
            target=self._jog_target,
            shoulder_pan_limits=tuple(self.arm_limits[0]),
            wrist_flex_limits=tuple(self.arm_limits[3]),
            wrist_roll_limits=tuple(self.arm_limits[4]),
            dt=self._jog_resolver.dt,
            gripper=gripper,
        )

    def reset(self, seed: int | None = None) -> Observation:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.reset_simulation()
        self.data.qpos[self.arm_qadr] = HOME_QPOS
        self.data.qpos[self.grip_qadr] = self.gripper_to_joint(1.0)
        self._jog_target = HOME_QPOS.copy()

        protected_xy = self.layout.reset(self.data, self.rng, self.cfg)
        self.randomization_metadata = self.randomization.apply(
            self.rng,
            seed=seed if seed is not None else self.cfg.seed,
            protected_xy=protected_xy,
        )

        self.data.ctrl[self.arm_act_ids] = HOME_QPOS
        self.data.ctrl[self.grip_act_id] = self.gripper_to_joint(1.0)
        mujoco.mj_forward(self.model, self.data)
        self._last_action = Action(
            joint_position=HOME_QPOS.copy(),
            gripper=GripperCommand(position=1.0),
        )
        return self.observe()

    def send_action(self, action: Action) -> None:
        if action.joint_position is None:
            raise ValueError(
                "Action.joint_position is required by this env. Convert a Twist "
                "command with physai.control.TwistToJointResolver first."
            )
        q_arm = np.clip(
            action.joint_position.reshape(5),
            self.arm_limits[:, 0],
            self.arm_limits[:, 1],
        )
        self.data.ctrl[self.arm_act_ids] = q_arm
        gripper = action.gripper or GripperCommand()
        self.data.ctrl[self.grip_act_id] = self.gripper_to_joint(gripper.clipped())

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict]:
        self.send_action(action)
        self.step_simulation()
        self._last_action = action
        observation = self.observe()
        info: dict = {}
        reward = 0.0
        terminated = False
        truncated = self.step_count >= self.cfg.max_steps
        info["randomization"] = self.randomization_metadata.as_dict()
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        super().close()

    def joint_state(self) -> JointState:
        position = np.concatenate(
            [
                self.data.qpos[self.arm_qadr],
                [self.data.qpos[self.grip_qadr]],
            ]
        )
        velocity = np.concatenate(
            [
                self.data.qvel[self.arm_vadr],
                [self.data.qvel[self.grip_vadr]],
            ]
        )
        effort = np.concatenate(
            [
                self.data.actuator_force[self.arm_act_ids],
                [self.data.actuator_force[self.grip_act_id]],
            ]
        )
        return JointState(
            name=ALL_JOINT_NAMES,
            position=position,
            velocity=velocity,
            effort=effort,
            header=Header(stamp=float(self.data.time), frame_id="base"),
        )

    def render_camera(self, name: str) -> np.ndarray:
        if not self.render_enabled:
            raise RuntimeError("env constructed with render=False")
        return super().render_camera(name)

    def observe(self) -> Observation:
        images: dict[str, ImageFrame] = {}
        if (
            self.render_enabled
            and self.cfg.camera_stride > 0
            and self.step_count % self.cfg.camera_stride == 0
        ):
            for camera in self.cfg.cameras:
                images[camera] = ImageFrame(
                    data=self.render_camera(camera),
                    camera_name=camera,
                    header=Header(
                        stamp=float(self.data.time),
                        frame_id=f"camera_{camera}",
                    ),
                )
        return Observation(
            joint_state=self.joint_state(),
            images=images,
            ee_pose=self.kin.fk(self.data),
            step=self.step_count,
            sim_time=float(self.data.time),
        )

    @property
    def target_color(self) -> str | None:
        """The color the current episode asks for (sorting scenes only)."""
        return self.layout.target_color

    @property
    def cube_pos(self) -> np.ndarray:
        return self.layout.cube_pos(self.data)

    @property
    def cube_geom_id(self) -> int:
        """Collision geom of the cube the task is currently asking for."""
        return self.layout.cube_geom_id()

    @property
    def cube_positions(self) -> dict[str, np.ndarray]:
        return self.layout.cube_positions(self.data)

    @property
    def target_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.target_sid].copy()

    @property
    def ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.kin.site_id].copy()

    @property
    def table_top(self) -> float:
        return self.cfg.scene.table_pos[2] + self.cfg.scene.table_size[2]

    @property
    def cube_half(self) -> float:
        return self.cfg.scene.cube_half
