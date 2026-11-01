"""ViperX 300 S (ALOHA follower arm) environment on the shared MuJoCo core.

The robot-facing surface is the SO-101's (`observe`, `step`, `reset`,
`robot_spec`, `cube_pos`, `target_pos`, `joint_to_gripper`, ...), so the web host,
the task runtime, the recorder and the scripted experts work on it unchanged. What
differs is the arm (six joints) and the gripper, whose prismatic finger is measured
in metres and normalised to 0 (closed) .. 1 (open) at the command boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property

import mujoco
import numpy as np

from ...contracts import (
    Action,
    CameraIntrinsics,
    GripperCommand,
    Header,
    ImageFrame,
    JointState,
    Observation,
    Pose,
    Quaternion,
    Vector3,
)
from ...robots.base import RobotSpec, RobotTrainingContract
from ...robots.reach import GraspProfile, GraspReach, grasp_reach_at

from ...sim.mujoco.core import MuJoCoSimulationCore
from ...sim.mujoco.domain_randomization import (
    DomainRandomizationConfig,
    DomainRandomizationEngine,
    RandomizationMetadata,
)
from ...sim.mujoco.scenes import (
    ManipulationSceneConfig,
    SingleCubePlaceSceneConfig,
)
from ..layout import create_layout
from .contracts import (
    ALL_JOINT_NAMES,
    ARM_JOINT_NAMES,
    viperx300s_training_contract,
)
from .kinematics import ViperX300SKinematics
from .scene import scene_defaults

# The Menagerie keyframe "home": arm folded up and forward, fingers open.
# How this arm grasps: the stock fingertips reach 0.036 m below the pinch site, so the
# pinch point is held 0.026 m above the cube centre; the wrist is pinned top-down.
VIPERX300S_GRASP = GraspProfile(
    grasp_dz=0.026,
    hover=0.08,
    carry=0.05,
    place_hover_dz=0.09,
    release_dz=0.03,
    grip_open=1.0,
    grip_hold=0.0,
    approach_dir=(0.0, 0.0, -1.0),
    open_descent=True,
    ik_kwargs={"max_iters": 400},
)

HOME_QPOS = np.array([0.0, -0.96, 1.16, 0.0, -0.3, 0.0], dtype=np.float64)

# MJCF names. The robot exposes the finger as `gripper`; the model calls the driven
# joint `left_finger` (the right finger follows it through an equality constraint)
# and the actuator `gripper`.
_LEFT_FINGER = "left_finger"
_RIGHT_FINGER = "right_finger"
_GRIPPER_ACTUATOR = "gripper"
_ARM_BODIES = (
    "base_link",
    "shoulder_link",
    "upper_arm_link",
    "upper_forearm_link",
    "lower_forearm_link",
    "wrist_link",
    "gripper_link",
    "gripper_prop_link",
    "left_finger_link",
    "right_finger_link",
)

# Where the cube may start. The arm's top-down reach at table height is roughly
# 0.20 .. 0.55 m from the base.
DEFAULT_CUBE_X_RANGE: tuple[float, float] = (0.28, 0.38)
DEFAULT_CUBE_Y_RANGE: tuple[float, float] = (0.05, 0.15)
DEFAULT_TARGET_X_RANGE: tuple[float, float] = (0.26, 0.40)
DEFAULT_TARGET_Y_RANGE: tuple[float, float] = (-0.16, -0.06)


@dataclass
class EnvConfig:
    """ViperX 300 S-specific simulation and observation settings."""

    scene: ManipulationSceneConfig = field(
        default_factory=lambda: SingleCubePlaceSceneConfig(**scene_defaults())
    )
    control_hz: float = 30.0
    render: bool = True
    cameras: tuple[str, ...] = ("front", "wrist")
    camera_stride: int = 1
    max_steps: int = 600
    randomize_cube: bool = True
    cube_x_range: tuple[float, float] = DEFAULT_CUBE_X_RANGE
    cube_y_range: tuple[float, float] = DEFAULT_CUBE_Y_RANGE
    randomize_target: bool = False
    target_x_range: tuple[float, float] = DEFAULT_TARGET_X_RANGE
    target_y_range: tuple[float, float] = DEFAULT_TARGET_Y_RANGE
    spawn_radius_range: tuple[float, float] | None = None
    min_cube_target_distance: float = 0.0
    seed: int | None = None
    domain_randomization: DomainRandomizationConfig = field(
        default_factory=DomainRandomizationConfig
    )
    # Optional cap (N) on the finger actuator. The model's own limit is 35 N; with
    # kp = 300 N/m a squeeze 5 mm past contact is already only ~1.5 N, so the default
    # leaves the model as shipped.
    gripper_force_limit: float | None = None
    # The Menagerie arm is driven by soft position actuators (kp 8..106 Nm/rad) that
    # sag by centimetres under the arm's own weight. The real ViperX's Dynamixel
    # servos hold their position stiffly, so by default gravity is cancelled on the
    # arm's bodies (not the cube) and the actuators only have to follow commands.
    gravity_compensation: bool = True


class ViperX300SEnv(MuJoCoSimulationCore):
    """ViperX 300 S embodiment environment for registered task scenes."""

    def __init__(self, cfg: EnvConfig | None = None) -> None:
        self.cfg = cfg or EnvConfig()
        defaults = scene_defaults()
        missing = {
            key: value
            for key, value in defaults.items()
            if getattr(self.cfg.scene, key) is None
        }
        if missing:
            self.cfg = replace(self.cfg, scene=replace(self.cfg.scene, **missing))
        self.model, self.spec = self.cfg.scene.build_model()
        # The arm's position actuators are soft (kp 8..106) and damped: with the default Euler
        # integrator the damping is explicit and the held arm trembles, which shakes a held
        # cube loose (1.5-3 mm in 60 s). Implicit-fast integrates it stably (0.03 mm), as
        # Newton's solver does.
        self.spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        # MuJoCo's friction is soft and the stock gripper squeezes a cube with only about 1 N per
        # side, so a held cube creeps unless friction is stiffer (impratio) and the contact is
        # damped (solref damping ratio 3, on the pads and the cube, whose values MuJoCo averages):
        # 3 mm in 60 s without, under 1 mm with.
        self.spec.option.impratio = 30.0
        for geom in self.spec.geoms:
            if "finger_pad" in geom.name or geom.name == "cube_geom":
                geom.solref = [0.01, 3.0]
        if self.cfg.gravity_compensation:
            # Set on the spec and recompile: MuJoCo skips gravity compensation unless
            # the compiled model counts at least one body that has it.
            for name in _ARM_BODIES:
                self.spec.body(name).gravcomp = 1.0
        self.model = self.spec.compile()
        self.randomization = DomainRandomizationEngine(
            self.model, self.cfg.domain_randomization
        )
        super().__init__(
            self.model,
            control_hz=self.cfg.control_hz,
            render=self.cfg.render,
            camera_width=self.cfg.scene.camera_width,
            camera_height=self.cfg.scene.camera_height,
            video_view=self.cfg.scene.video_view,
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
        self.gripper_joint_id = jid(_LEFT_FINGER)
        right_joint_id = jid(_RIGHT_FINGER)
        self.arm_qadr = self.model.jnt_qposadr[self.arm_joint_ids]
        self.arm_vadr = self.model.jnt_dofadr[self.arm_joint_ids]
        self.grip_qadr = int(self.model.jnt_qposadr[self.gripper_joint_id])
        self.grip_vadr = int(self.model.jnt_dofadr[self.gripper_joint_id])
        self.right_grip_qadr = int(self.model.jnt_qposadr[right_joint_id])
        self.arm_act_ids = np.array([aid(name) for name in ARM_JOINT_NAMES])
        self.grip_act_id = aid(_GRIPPER_ACTUATOR)
        self.arm_limits = self.model.jnt_range[self.arm_joint_ids].copy()
        self.grip_limits = self.model.jnt_range[self.gripper_joint_id].copy()
        if self.cfg.gripper_force_limit is not None:
            limit = float(self.cfg.gripper_force_limit)
            self.model.actuator_forcerange[self.grip_act_id] = [-limit, limit]

        self.layout = create_layout(self.model, self.cfg.scene)
        self.target_sid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "target_site"
        )
        self.kin = ViperX300SKinematics(
            self.model, ee_site=self.cfg.scene.description.ee_site
        )
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
            name="viperx300s",
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
            # Same reasoning as the SO-101: the gate compares a command against the
            # measured position, which trails it, so leave headroom for servo lag.
            max_joint_delta={name: 0.75 for name in ARM_JOINT_NAMES},
            metadata={
                "control_hz": self.cfg.control_hz,
                "action_schema": "viperx300s.joint_position.v1",
                "simulator": "mujoco",
                "gripper_unit": "m",
            },
            joint_state_frame="base",
            camera_frames={"front": "camera_front", "wrist": "camera_wrist"},
            # The core requires "rad" and "rad/s" here. The gripper entry of
            # `joint_state` is the exception: a finger position in metres
            # (see `metadata["gripper_unit"]`).
            units={
                "joint_position": "rad",
                "joint_velocity": "rad/s",
                "position": "m",
            },
        )

    @property
    def training_contract(self) -> RobotTrainingContract:
        """Return the configured ViperX 300 S contract for training adapters."""
        camera_config = {
            name: {
                "width": self.cfg.scene.camera_width,
                "height": self.cfg.scene.camera_height,
                "encoding": "rgb8",
            }
            for name in self.cfg.cameras
        }
        return viperx300s_training_contract(camera_config=camera_config)

    def gripper_to_joint(self, normalized: float) -> float:
        """Finger position (m) for a command in 0 (closed) .. 1 (open)."""
        lo, hi = self.grip_limits
        return float(lo + np.clip(normalized, 0.0, 1.0) * (hi - lo))

    def joint_to_gripper(self, q: float) -> float:
        lo, hi = self.grip_limits
        return float(np.clip((q - lo) / (hi - lo), 0.0, 1.0))

    def _set_fingers(self, position: float) -> None:
        self.data.qpos[self.grip_qadr] = position
        self.data.qpos[self.right_grip_qadr] = -position

    def reset(self, seed: int | None = None) -> Observation:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.reset_simulation()
        self.data.qpos[self.arm_qadr] = HOME_QPOS
        self._set_fingers(self.gripper_to_joint(1.0))

        protected_xy = self.layout.reset(self.data, self.rng, self.cfg)
        self.randomization_metadata = self.randomization.apply(
            self.rng,
            seed=seed if seed is not None else self.cfg.seed,
            protected_xy=protected_xy,
        )
        self.layout.show_target()

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
            action.joint_position.reshape(len(ARM_JOINT_NAMES)),
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

    def _finger_pads(self) -> set[int]:
        return {
            geom
            for geom in range(self.model.ngeom)
            if (
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
            ).startswith(("left_finger_pad_", "right_finger_pad_"))
        }

    def gripper_contact_force(self) -> float:
        """Total normal force (N) on the two fingers' pads right now."""
        pads = self._finger_pads()
        wrench, total = np.zeros(6), 0.0
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if contact.geom1 in pads or contact.geom2 in pads:
                mujoco.mj_contactForce(self.model, self.data, index, wrench)
                total += float(wrench[0])
        return total

    def joint_state(self) -> JointState:
        position = np.concatenate(
            [self.data.qpos[self.arm_qadr], [self.data.qpos[self.grip_qadr]]]
        )
        velocity = np.concatenate(
            [self.data.qvel[self.arm_vadr], [self.data.qvel[self.grip_vadr]]]
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

    def camera_calibration(self, name: str) -> tuple[CameraIntrinsics, Pose]:
        """This camera's pinhole intrinsics and its pose in the base frame.

        The pose uses the ROS optical-frame convention (x right, y down, z forward),
        converted from MuJoCo's (x right, y up, z backward), as the SO-101's does.
        """
        camera_id = self.model.camera(name).id
        width, height = self.cfg.scene.camera_width, self.cfg.scene.camera_height
        fy = height / (
            2.0 * np.tan(np.deg2rad(float(self.model.cam_fovy[camera_id])) / 2.0)
        )
        intrinsics = CameraIntrinsics(
            fx=fy, fy=fy, cx=(width - 1) / 2.0, cy=(height - 1) / 2.0
        )
        rotation = self.data.cam_xmat[camera_id].reshape(3, 3) @ np.diag(
            [1.0, -1.0, -1.0]
        )
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rotation.reshape(9))
        position = self.data.cam_xpos[camera_id].copy()
        if not self.cfg.domain_randomization.camera_shift_calibrated:
            body_rotation = self.data.xmat[self.model.cam_bodyid[camera_id]]
            position -= body_rotation.reshape(3, 3) @ self.randomization.camera_shift(
                camera_id
            )
        extrinsics = Pose(
            position=Vector3.from_array(position),
            orientation=Quaternion.from_mujoco(quat),
        )
        return intrinsics, extrinsics

    def observe(self) -> Observation:
        images: dict[str, ImageFrame] = {}
        if (
            self.render_enabled
            and self.cfg.camera_stride > 0
            and self.step_count % self.cfg.camera_stride == 0
        ):
            for camera in self.cfg.cameras:
                intrinsics, extrinsics = self.camera_calibration(camera)
                images[camera] = ImageFrame(
                    data=self.render_camera(camera),
                    camera_name=camera,
                    header=Header(
                        stamp=float(self.data.time), frame_id=f"camera_{camera}"
                    ),
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
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
        return self.layout.target_color

    @property
    def cube_pos(self) -> np.ndarray:
        return self.layout.cube_pos(self.data)

    @property
    def cube_geom_id(self) -> int:
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

    @property
    def rest_z(self) -> float:
        """Height a cube resting on the table sits at."""
        return self.table_top + self.cube_half

    def pinch_center(self) -> np.ndarray:
        """Where the gripper's pinch point currently is."""
        return self.kin.pinch_center(self.data)

    def grasp_profile(self) -> GraspProfile:
        """How this arm grasps a cube."""
        return VIPERX300S_GRASP

    def grasp_reach_defaults(self) -> dict[str, float]:
        """Pinch height, hover clearance and carry lift `grasp_reach` uses unless told."""
        return self.grasp_profile().reach_defaults(self.rest_z)

    def grasp_reach(
        self,
        x: float,
        y: float,
        *,
        z: float | None = None,
        hover: float | None = None,
        carry: float | None = None,
        tilt: float = 90.0,
        q_init: np.ndarray | None = None,
    ) -> GraspReach:
        """Whether the arm grasps at (x, y) top-down, fingers closing along the world y axis.

        This is the grasp the expert asks for, so `tilt` can only be 90. `q_init` is a
        joint-state position vector to start the IK from (default: the kinematics' own)."""
        if tilt != 90.0:
            raise ValueError("the ViperX 300 S grasps top-down only (tilt 90)")
        return grasp_reach_at(
            self.kin,
            self.grasp_reach_defaults(),
            x,
            y,
            np.array([0.0, 0.0, -1.0]),
            z=z,
            hover=hover,
            carry=carry,
            q_init=q_init,
            max_iters=400,
        )
