"""SO-101 robot environment backed by Isaac Sim (`physai.sim.isaac`).

A robot-owned backend, like `SO101Env`: observation, action, and lifecycle
only, no task (`TaskRuntime` wraps either backend the same way). Only this
module and `physai.sim.isaac` may import `isaacsim`/`omni`/`pxr`
(`pyproject.toml`'s import-linter contracts enforce this), so nothing above
the `RobotPort` boundary needs Isaac installed to import `physai.robots`.

Deliberately smaller than `SO101Env`: no Cartesian jog, and a scene of one
cube only (`cfg.scene`, built by `sim.isaac.scene.add_workspace` from the same
`WorkspaceConfig` MuJoCo reads). `self.kin` is a real `ArmKinematics`, backed by a MuJoCo
model loaded purely as an FK/IK math tool and never simulated — Isaac's own
physics runs everything; this only reuses already-verified kinematics math
instead of reimplementing it against USD/PhysX (importing `mujoco` here is
fine: only `isaacsim`/`omni`/`pxr` are import-linter-restricted to this
module and `physai.sim.isaac`, and `mujoco` is a base dependency regardless).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import mujoco
import numpy as np

from ...contracts import (
    DEFAULT_CAMERA_RESOLUTION,
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
    parse_camera_resolution,
)
from ...sim.isaac.core import IsaacSimulationCore, ensure_simulation_app
from ...sim.isaac.description import (
    apply_actuators,
    apply_cameras,
    apply_contact_friction,
    apply_contact_pad_colliders,
    apply_frames,
)
from ...sim.isaac.description import import_robot as isaac_import_robot
from ...sim.isaac.scene import (
    add_ground_plane,
    add_studio_lighting,
    add_workspace,
    checker_texture,
)
from ...sim.mujoco.scenes import ManipulationSceneConfig
from ...sim.workspace import FRONT_CAMERA_FOVY_DEG, CubeSpec
from ..base import RobotSpec, RobotTrainingContract
from ..description import RobotDescription, load_robot_description
from .contracts import (
    ALL_JOINT_NAMES,
    ARM_JOINT_NAMES,
    GRIPPER_JOINT_NAME,
    so101_training_contract,
)
from .kinematics import ArmKinematics
from .scene import scene_defaults
from .layout import (
    DEFAULT_CUBE_X_RANGE,
    DEFAULT_CUBE_Y_RANGE,
    DEFAULT_TARGET_X_RANGE,
    DEFAULT_TARGET_Y_RANGE,
    draw_cube_xy,
    draw_target_xy,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
_DESCRIPTION_PATH = Path(__file__).resolve().parent / "description.yaml"
HOME_QPOS = np.array([0.0, -1.05, 1.25, 0.75, 0.0], dtype=np.float64)
# Mean absolute pixel difference between a render with and without the robot
# below which the robot is taken to be missing. Measured on the front camera:
# a drawn robot gives ~10, two renders of the same state differ by ~1.3 (the
# denoiser), and a scene without the robot by ~0.3.
_ROBOT_RENDER_MIN_DIFF = 3.0


@dataclass
class IsaacEnvConfig:
    """SO-101-on-Isaac simulation and observation settings.

    Narrower than `so101.mujoco_env.EnvConfig`: single-cube scenes only, a
    fixed target, and only two difficulty knobs (`lighting_scale`,
    `camera_position_jitter`); Isaac's own randomization tooling is a separate
    integration. Give it the same `scene` MuJoCo uses and the table, cube,
    target, front camera and per-seed cube layout follow from it; without one
    the env holds the robot alone.
    """

    scene: ManipulationSceneConfig | None = None
    description: RobotDescription | None = None
    assets_root: Path = field(default_factory=lambda: REPO_ROOT / "assets" / "so101")
    usd_out_dir: Path = field(
        default_factory=lambda: REPO_ROOT / ".isaac_cache" / "so101"
    )
    # One of `physai.contracts.CAMERA_RESOLUTIONS`; every camera renders at it.
    camera_resolution: str = DEFAULT_CAMERA_RESOLUTION
    physics_variant: str = "physx"
    control_hz: float = 30.0
    physics_hz: float = 60.0
    render: bool = True
    headless: bool = True
    cameras: tuple[str, ...] = ("wrist",)
    max_steps: int = 400
    seed: int | None = None
    gripper_force_limit: float = 0.3
    # Per-seed cube placement, as MuJoCo's `EnvConfig`: None means "randomize
    # when a scene is given" (the MuJoCo default), otherwise the cube stays at
    # its configured position.
    randomize_cube: bool | None = None
    cube_x_range: tuple[float, float] = DEFAULT_CUBE_X_RANGE
    cube_y_range: tuple[float, float] = DEFAULT_CUBE_Y_RANGE
    randomize_target: bool = False
    target_x_range: tuple[float, float] = DEFAULT_TARGET_X_RANGE
    target_y_range: tuple[float, float] = DEFAULT_TARGET_Y_RANGE
    # Study 1's reachable region: when set, cube and target are redrawn until
    # their distance from the robot base lies in this range (metres), and the
    # target until it is `min_cube_target_distance` from the cube.
    spawn_radius_range: tuple[float, float] | None = None
    min_cube_target_distance: float = 0.0
    # Difficulty knobs, named as in MuJoCo's `DomainRandomizationConfig`.
    # `lighting_scale` multiplies the dome and key light. `camera_position_jitter`
    # moves every camera by a uniform offset of up to this many metres per
    # axis each episode (drawn from the env's seeded rng after the cube) while
    # `camera_calibration` keeps reporting the nominal pose, the equivalent of
    # MuJoCo's `camera_shift_calibrated=False`.
    lighting_scale: float = 1.0
    camera_position_jitter: float = 0.0

    def __post_init__(self) -> None:
        parse_camera_resolution(self.camera_resolution)
        if self.lighting_scale <= 0:
            raise ValueError("lighting_scale must be positive")
        if self.camera_position_jitter < 0:
            raise ValueError("camera_position_jitter must be non-negative")
        if self.scene is not None:
            if self.scene.camera_resolution != self.camera_resolution:
                raise ValueError(
                    f"scene camera_resolution {self.scene.camera_resolution!r} "
                    f"differs from the env's {self.camera_resolution!r}"
                )
            cube_names = [cube.name for cube in self.scene.cubes()]
            if len(cube_names) > 1:
                raise ValueError(
                    "SO101IsaacEnv supports single-cube scenes only; "
                    f"this scene has cubes {tuple(cube_names)}"
                )
        if self.randomize_cube is None:
            self.randomize_cube = self.scene is not None
        self.cube_x_range = tuple(self.cube_x_range)
        self.cube_y_range = tuple(self.cube_y_range)
        self.target_x_range = tuple(self.target_x_range)
        self.target_y_range = tuple(self.target_y_range)
        if self.spawn_radius_range is not None:
            self.spawn_radius_range = tuple(self.spawn_radius_range)
        self.cameras = tuple(self.cameras)


class SO101IsaacEnv:
    """SO-101 embodiment environment backed by Isaac Sim's PhysX/Newton solver."""

    def __init__(self, cfg: IsaacEnvConfig | None = None) -> None:
        self.cfg = cfg or IsaacEnvConfig()
        if getattr(self.cfg.scene, "floor_style", "checker") != "checker":
            raise ValueError("Isaac Sim only draws the checker floor (floor_style)")
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
        # Display/telemetry mirror for `physai.web.Host` (`--serve`): never
        # simulated, only refreshed from Isaac's joint state (and the cube's
        # pose) by `observe()`. With a scene it holds the table, the cube and
        # the target too, as the MuJoCo backend's model does.
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

            # Built once physics is initialized (the tensor backend needs it);
            # reads and teleports go through it so they never see a stale USD.
            self._cube_body = RigidPrim(self._cube_path)

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
        if self.core.render_enabled and self._camera_prims:
            self._require_rendered_robot()

    def _build_kinematics_oracle(self) -> ArmKinematics:
        """A MuJoCo model used purely as an FK/IK math tool, never simulated.

        Loads the same MJCF the MuJoCo backend simulates (so joint and site
        names match), only to reuse `ArmKinematics`'s existing FK/IK math
        instead of reimplementing it against USD/PhysX. Isaac's own physics
        (`self.articulation`, `self.core`) is what actually runs the robot.
        With a scene the model is that scene's, built as `SO101Env` builds it,
        so the viewer mirror shows the table, the cube and the target.
        """
        if self.cfg.scene is not None:
            defaults = scene_defaults()
            missing = {
                key: value
                for key, value in defaults.items()
                if getattr(self.cfg.scene, key) is None
            }
            scene = replace(self.cfg.scene, **missing) if missing else self.cfg.scene
            model, _ = scene.build_model()
            return ArmKinematics(model)
        mjcf_path = self.cfg.assets_root / self.description.mjcf
        model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        return ArmKinematics(model)

    def _cube_spec(self) -> CubeSpec:
        cubes = () if self.cfg.scene is None else self.cfg.scene.cubes()
        if not cubes:
            raise AttributeError(
                "this SO101IsaacEnv has no cube (its scene places none)"
            )
        return cubes[0]

    @property
    def cube_pos(self) -> np.ndarray:
        if self._cube_path is None:
            raise AttributeError(
                "this SO101IsaacEnv has no cube (its scene places none)"
            )
        positions, _ = self._cube_body.get_world_poses()
        return np.asarray(positions, dtype=np.float64)[0]

    @property
    def table_top(self) -> float:
        """Height of the surface objects rest on (the ground when no scene)."""
        if self.cfg.scene is None:
            return 0.0
        return self.cfg.scene.table_pos[2] + self.cfg.scene.table_size[2]

    @property
    def cube_half(self) -> float:
        return self._cube_spec().half_size

    @property
    def rest_z(self) -> float:
        """Height a cube rests at: the table top plus the cube's half size.
        `SO101VisualServoPolicy`'s `_waypoint()` reads this where MuJoCo's
        `cfg.scene.table_pos`/`table_size`/`cube_half` would be."""
        return self.table_top + self.cube_half

    @property
    def target_pos(self) -> np.ndarray:
        if self.cfg.scene is None:
            raise AttributeError("this SO101IsaacEnv has no target (it has no scene)")
        return self._target_pos.copy()

    def camera_calibration(self, name: str) -> tuple[CameraIntrinsics, Pose]:
        """This camera's pinhole intrinsics and its pose in the base frame.

        `"front"` is world-fixed (the scene's `front_cam_*`). `"wrist"` (any
        description camera) is link-mounted, so its pose is the parent
        link's current pose from the display mirror (`self.data`, refreshed
        by `observe()` before it asks) composed with the description's
        mount -- the same chain MuJoCo's own `cam_xmat` resolves. This is
        what lets `SO101VisualServoPolicy._refine_from_final_camera` re-aim
        the grasp from the wrist view during DESCEND; without it the grasp
        keeps the front camera's centimetre-level triangulation error.
        """
        if name == "front" and self.cfg.scene is not None:
            scene = self.cfg.scene
            width, height = self.core.camera_size
            fovy_deg = FRONT_CAMERA_FOVY_DEG
            x_axis = np.asarray(scene.front_cam_xyaxes[:3], dtype=np.float64)
            y_axis = np.asarray(scene.front_cam_xyaxes[3:], dtype=np.float64)
            x_axis /= np.linalg.norm(x_axis)
            y_axis /= np.linalg.norm(y_axis)
            z_axis = np.cross(x_axis, y_axis)
            # MuJoCo's own camera axes: x right, y up, z backward.
            rotation = np.stack([x_axis, y_axis, z_axis], axis=1)
            position = np.asarray(scene.front_cam_pos, dtype=np.float64)
        else:
            mount = self._mounted_camera(name)
            if mount is None:
                raise KeyError(f"camera {name!r} has no calibration in this env")
            camera, body_id = mount
            # The render size is the chosen resolution, not the description's
            # own (320x240) default; the field of view is the mount's.
            width, height = self._render_size
            fovy_deg = camera.fovy_deg
            mount_rotation = np.zeros(9)
            mujoco.mju_quat2Mat(
                mount_rotation, np.asarray(camera.quat, dtype=np.float64)
            )
            body_rotation = self.data.xmat[body_id].reshape(3, 3)
            rotation = body_rotation @ mount_rotation.reshape(3, 3)
            position = self.data.xpos[body_id] + body_rotation @ np.asarray(
                camera.pos, dtype=np.float64
            )
        # A pinhole camera's focal length is one physical quantity; fx == fy
        # in pixel units once the horizontal aperture is itself scaled by
        # width/height (`add_world_camera`'s `CreateHorizontalApertureAttr`
        # call does this) -- an `fx = fy * width / height` cross term here
        # double-counts that scaling and mis-triangulates any pixel whose
        # horizontal axis isn't aligned with a world axis this ratio cancels
        # out for by symmetry (verified against the tier-4 visual-servo
        # test: this camera's horizontal pixel axis maps to world Y, and a
        # 4:3 aspect miscalibration here showed up as a ~2cm world-Y offset
        # between the triangulated cube position and its real position).
        fy = height / (2.0 * np.tan(np.deg2rad(fovy_deg) / 2.0))
        intrinsics = CameraIntrinsics(
            fx=fy, fy=fy, cx=(width - 1) / 2.0, cy=(height - 1) / 2.0
        )
        # MuJoCo's camera axes (x right, y up, z backward) -> the ROS
        # optical-frame convention ImageFrame.extrinsics documents (x right,
        # y down, z forward); mirrors mujoco_env.py's camera_calibration().
        optical = rotation @ np.diag([1.0, -1.0, -1.0])
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, optical.reshape(9))
        extrinsics = Pose(
            position=Vector3.from_array(position),
            orientation=Quaternion.from_mujoco(quat),
        )
        return intrinsics, extrinsics

    def _mounted_camera(self, name: str):
        """`(CameraDescription, mirror body id)` for a link-mounted camera
        whose parent link exists in the mirror model, else None. Like the USD
        side (`apply_cameras`), the first description entry that matches wins.
        """
        for camera in self.description.cameras:
            if camera.name != name:
                continue
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, camera.parent_link
            )
            if body_id >= 0:
                return camera, body_id
        return None

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
            # From the kinematics model (float64), like MuJoCo's spec and the
            # IK solutions the safety gate validates; Isaac's own limits are
            # float32 and a solution exactly at a limit would exceed them by
            # rounding. `send_action` clips to Isaac's physical limits.
            joint_limits={
                name: (float(lo), float(hi))
                for name, (lo, hi) in zip(ARM_JOINT_NAMES, self.kin.limits)
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
        cube_xy = self._reset_cube()
        if self.cfg.randomize_target:
            self._reset_target(cube_xy)
        self._shift_cameras()
        target = np.zeros((1, len(self.dof_names)), dtype=np.float32)
        target[0, self._arm_indices] = HOME_QPOS
        target[0, self._gripper_index] = self.gripper_to_joint(1.0)
        self.articulation.set_dof_position_targets(target)
        self.articulation.set_dof_positions(target)
        self._last_action = Action(
            joint_position=HOME_QPOS.copy(), gripper=GripperCommand(position=1.0)
        )
        # Poses written through the tensor API reach the renderer only after a
        # physics step; without one, the first images show the previous episode
        # (a cube still on the target pad), and a camera policy aims at that.
        # Hold at HOME for one control step so the first observation is fresh.
        self.core.step_simulation()
        self.core.step_count = 0
        return self.observe()

    def _shift_cameras(self) -> None:
        """Move every camera prim off its nominal mount by this episode's jitter.

        The nominal translate is read once, the first time it is needed, so
        episodes shift from the mount and not from each other's offsets.
        """
        jitter = self.cfg.camera_position_jitter
        if jitter <= 0:
            return
        from pxr import Gf, UsdGeom

        if not hasattr(self, "_camera_mounts"):
            self._camera_mounts = {}
            for name, path in self._camera_prims.items():
                op = UsdGeom.Xformable(
                    self.stage.GetPrimAtPath(path)
                ).GetOrderedXformOps()[0]
                self._camera_mounts[name] = (op, np.array(op.Get(), dtype=np.float64))
        for name in sorted(self._camera_mounts):
            op, nominal = self._camera_mounts[name]
            op.Set(Gf.Vec3d(*(nominal + self.rng.uniform(-jitter, jitter, size=3))))

    def _reset_target(self, cube_xy: tuple[float, float] | None) -> None:
        """Move the target pad to this episode's draw, as MuJoCo's layout does:
        right after the cube, from the same rng, kept clear of the cube."""
        from pxr import Gf, UsdGeom

        self._target_pos = self._target_pos.copy()
        self._target_pos[:2] = draw_target_xy(
            self.rng, self.cfg, [cube_xy] if cube_xy is not None else []
        )
        op = UsdGeom.Xformable(
            self.stage.GetPrimAtPath(self._target_pad_path)
        ).GetOrderedXformOps()[0]
        op.Set(Gf.Vec3d(*self._target_pos))

    def _reset_cube(self) -> tuple[float, float] | None:
        """Put the cube at this episode's starting pose, at rest.

        `reset_simulation()` re-initializes physics but leaves a rigid body
        wherever the previous episode flung it, so a second episode would
        start with the cube off the table. The pose is the configured one, or
        with `randomize_cube` the same per-seed draw MuJoCo makes
        (`layout.draw_cube_xy` on `self.rng`) at the configured height.
        """
        if self._cube_path is None:
            return None
        position = list(self._cube_spec().position)
        if self.cfg.randomize_cube:
            position[:2] = draw_cube_xy(self.rng, self.cfg)
        self._cube_body.set_world_poses(
            positions=[position], orientations=[[1.0, 0.0, 0.0, 0.0]]
        )
        self._cube_body.set_velocities(
            linear_velocities=[[0.0, 0.0, 0.0]], angular_velocities=[[0.0, 0.0, 0.0]]
        )
        return position[0], position[1]

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

    def robot_is_rendered(self) -> bool:
        """Whether the cameras actually draw the robot.

        Now and then Isaac Sim starts with the robot missing from every render
        (no arm, no shadow) while its physics works, so camera policies run
        blind and fail. The test needs no knowledge of colours: render the
        first camera with the robot shown and with it hidden; if the images
        do not differ, the robot was never drawn.
        """
        from pxr import UsdGeom

        app = ensure_simulation_app()
        camera = next(iter(self._camera_prims))
        root = UsdGeom.Imageable(self.stage.GetPrimAtPath(self.robot_prim_path))

        def snapshot() -> np.ndarray:
            for _ in range(3):
                app.update()
            for _ in range(4):  # the denoiser keeps ghosts of the previous state
                self.render_camera(camera)
            return self.render_camera(camera).astype(np.float64)

        shown = snapshot()
        root.MakeInvisible()
        try:
            hidden = snapshot()
        finally:
            root.MakeVisible()
            snapshot()
        return float(np.abs(shown - hidden).mean()) > _ROBOT_RENDER_MIN_DIFF

    def _require_rendered_robot(self) -> None:
        # No in-process recovery: toggling the visual geometry invalidates the
        # articulation (its links are that geometry's parents), and nothing
        # else was found to bring the robot back; a new process does.
        if not self.robot_is_rendered():
            raise RuntimeError(
                "Isaac Sim is not drawing the robot (a startup glitch that "
                "makes camera policies run blind); restart the process"
            )

    def render_camera(self, name: str) -> np.ndarray:
        prim_path = self._camera_prims.get(name)
        if prim_path is None:
            raise KeyError(f"camera {name!r} not attached to this description")
        return self.core.render_camera(prim_path)

    def _sync_mirror(self, joint_state: JointState) -> None:
        """Copy Isaac's joint positions into the MuJoCo display mirror."""
        by_name = dict(zip(joint_state.name, joint_state.position))
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

    def observe(self) -> Observation:
        joint_state = self._joint_state()
        self._sync_mirror(joint_state)
        images: dict[str, ImageFrame] = {}
        if self.core.render_enabled:
            for camera in self.cfg.cameras:
                if camera not in self._camera_prims:
                    continue
                try:
                    intrinsics, extrinsics = self.camera_calibration(camera)
                except KeyError:
                    intrinsics, extrinsics = None, None
                images[camera] = ImageFrame(
                    data=self.render_camera(camera),
                    camera_name=camera,
                    header=Header(
                        stamp=self.step_count * self.core.control_dt,
                        frame_id=f"camera_{camera}",
                    ),
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                )
        return Observation(
            joint_state=joint_state,
            images=images,
            step=self.step_count,
            sim_time=self.step_count * self.core.control_dt,
        )

    def send_action(self, action: Action) -> None:
        if action.joint_position is None:
            raise ValueError("SO101IsaacEnv requires joint-position actions")
        target = np.asarray(self.articulation.get_dof_position_targets()).copy()
        target[0, self._arm_indices] = np.clip(
            action.joint_position, self.arm_limits[:, 0], self.arm_limits[:, 1]
        )
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


__all__ = [
    "IsaacEnvConfig",
    "SO101IsaacEnv",
]
