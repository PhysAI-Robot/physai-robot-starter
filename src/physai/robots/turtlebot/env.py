"""MuJoCo environment backed by the TurtleBot4 MJCF model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from ...contracts import (
    Action,
    Header,
    ImageFrame,
    JointState,
    Observation,
    Pose,
    PoseStamped,
    Quaternion,
    Twist,
    Vector3,
)
from ..base import RobotSpec
from ...sim.core import MuJoCoSimulationCore

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL = REPO_ROOT / "assets" / "turtlebot4" / "turtlebot4.xml"
BASE_BODY = "base"
CHASE_CAMERA = "physai_chase"


@dataclass
class TurtleBot4Config:
    """Parameters for the TurtleBot4 MuJoCo model."""

    model_path: Path = DEFAULT_MODEL
    control_hz: float = 10.0
    max_steps: int = 500
    render: bool = False
    initial_pose: tuple[float, float, float] = (0.0, 0.0, 0.1)
    seed: int | None = None


def _compile_scene(model_path: Path) -> mujoco.MjModel:
    """Load the vendored TurtleBot4 MJCF and add what it needs to be usable.

    The upstream model ships a worldbody containing only the robot: no floor
    geom, no light, and no camera, while gravity stays enabled. Loaded as-is
    the base free-falls forever (z reaches -1900 m within a few hundred steps)
    and every rendered frame is black, so the advertised ``images`` capability
    returns nothing. A smoke test that only checks "stepping does not raise"
    does not catch either problem.

    The asset is fetched into the git-ignored ``assets/`` directory, so these
    additions are made here rather than by editing a third-party file that
    re-fetching would overwrite. Each one is skipped when the upstream model
    starts providing its own.
    """
    spec = mujoco.MjSpec.from_file(str(model_path))

    has_plane = any(
        geom.type == mujoco.mjtGeom.mjGEOM_PLANE
        for body in spec.bodies
        for geom in body.geoms
    )
    if not has_plane:
        # A checker material, matching the SO-101 scene: on a uniform plane
        # there is no visual reference, so a driving base looks stationary.
        spec.add_texture(
            name="physai_grid",
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            rgb1=[0.22, 0.24, 0.28],
            rgb2=[0.16, 0.18, 0.22],
            width=300,
            height=300,
        )
        spec.add_material(
            name="physai_grid",
            textures=["", "physai_grid"],
            texuniform=True,
            texrepeat=[6, 6],
            reflectance=0.1,
        )
        spec.worldbody.add_geom(
            name="physai_ground",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[0, 0, 0.05],
            pos=[0, 0, 0],
            material="physai_grid",
        )

    if not spec.lights:
        spec.worldbody.add_light(
            name="physai_light",
            pos=[0, 0, 3.0],
            dir=[0, 0, -1],
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            diffuse=[0.7, 0.7, 0.7],
        )

    if not spec.cameras:
        # Position tracks the base centre of mass while the orientation stays
        # fixed, so the robot stays framed wherever it drives.
        spec.worldbody.add_camera(
            name=CHASE_CAMERA,
            mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM,
            targetbody=BASE_BODY,
            pos=[-1.2, 0.0, 0.8],
            xyaxes=[0.0, -1.0, 0.0, 0.482, 0.0, 0.876],
            fovy=55,
        )

    return spec.compile()


class TurtleBot4Env(MuJoCoSimulationCore):
    """Gym-like wrapper around the credited TurtleBot4 MJCF model."""

    _WHEEL_NAMES = ("left_wheel", "right_wheel")

    def __init__(self, cfg: TurtleBot4Config | None = None) -> None:
        self.cfg = cfg or TurtleBot4Config()
        if self.cfg.control_hz <= 0:
            raise ValueError("control_hz must be positive")
        if not self.cfg.model_path.exists():
            raise FileNotFoundError(
                f"TurtleBot4 model missing: {self.cfg.model_path}. "
                "Run `python scripts/fetch_assets.py --robot turtlebot4`."
            )
        self.model = _compile_scene(self.cfg.model_path)
        self.model.opt.timestep = min(self.model.opt.timestep, 0.002)
        super().__init__(
            self.model,
            control_hz=self.cfg.control_hz,
            render=self.cfg.render,
            camera_width=640,
            camera_height=480,
        )
        self.rng = np.random.default_rng(self.cfg.seed)
        self._actuator_ids = {self.model.actuator(i).name: i for i in range(self.model.nu)}
        self._base_body_id = self.model.body(BASE_BODY).id
        self._has_chase_camera = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, CHASE_CAMERA
        ) >= 0

    @property
    def robot_spec(self) -> RobotSpec:
        return RobotSpec(
            name="turtlebot4",
            kind="mobile_base",
            joint_names=self._WHEEL_NAMES,
            action_joint_names=(),
            action_modes=("twist",),
            observation_modalities=("state", "images", "ee_pose"),
            capabilities=("base_velocity", "odometry", "images", "imu", "lidar"),
            metadata={"drive": "differential", "model": str(self.cfg.model_path)},
            joint_state_frame="base_link",
            action_frame="base",
            camera_frames={"free": "base_link"},
        )

    def reset(self, seed: int | None = None) -> Observation:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.reset_simulation()
        x, y, z = self.cfg.initial_pose
        free_qadr = int(self.model.jnt_qposadr[self.model.joint("floating_base_joint").id])
        self.data.qpos[free_qadr:free_qadr + 7] = (x, y, z, 1.0, 0.0, 0.0, 0.0)
        self.data.ctrl[:] = 0.0
        if "lidar_spin" in self._actuator_ids:
            self.data.ctrl[self._actuator_ids["lidar_spin"]] = 0.05
        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    def send_action(self, action: Action) -> None:
        if action.ee_twist is None:
            raise ValueError("Action.ee_twist is required by the turtlebot4 env")
        twist: Twist = action.ee_twist
        self.data.ctrl[self._actuator_ids["forward"]] = float(twist.linear.x)
        self.data.ctrl[self._actuator_ids["turn"]] = float(twist.angular.z)

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict]:
        self.send_action(action)
        self.step_simulation()
        info = {"pose": self._pose_array()}
        return self.observe(), 0.0, False, self.step_count >= self.cfg.max_steps, info

    def close(self) -> None:
        super().close()

    def joint_state(self) -> JointState:
        positions = np.array([self.data.joint(name).qpos[0] for name in ("left", "right")])
        velocities = np.array([self.data.joint(name).qvel[0] for name in ("left", "right")])
        return JointState(
            name=self._WHEEL_NAMES,
            position=positions,
            velocity=velocities,
            effort=np.zeros(2),
            header=Header(stamp=float(self.data.time), frame_id="base_link"),
        )

    def render_camera(self, name: str = "free") -> np.ndarray:
        if self._renderer is None:
            raise RuntimeError("env constructed with render=False")
        camera: str | int = name
        if name == "free":
            # MuJoCo's free camera is static, so the base drove out of frame
            # within a couple of seconds and the published image went blank.
            # `camera_frames` already declares this image to be in base_link,
            # so resolve it to the base-tracking camera when one is available.
            camera = CHASE_CAMERA if self._has_chase_camera else -1
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def observe(self) -> Observation:
        images = {}
        if self._renderer is not None:
            images["free"] = ImageFrame(
                data=self.render_camera(),
                camera_name="free",
                header=Header(stamp=float(self.data.time), frame_id="base_link"),
            )
        return Observation(
            joint_state=self.joint_state(),
            images=images,
            ee_pose=self._pose_stamped(),
            step=self.step_count,
            sim_time=float(self.data.time),
        )

    def _pose_array(self) -> np.ndarray:
        body = self.data.body(self._base_body_id)
        yaw = float(np.arctan2(2 * (body.xquat[0] * body.xquat[3] + body.xquat[1] * body.xquat[2]),
                               1 - 2 * (body.xquat[2] ** 2 + body.xquat[3] ** 2)))
        return np.array([body.xpos[0], body.xpos[1], yaw])

    def _pose_stamped(self) -> PoseStamped:
        body = self.data.body(self._base_body_id)
        return PoseStamped(
            pose=Pose(
                position=Vector3.from_array(body.xpos),
                orientation=Quaternion.from_mujoco(body.xquat),
            ),
            header=Header(stamp=float(self.data.time), frame_id="odom"),
        )
