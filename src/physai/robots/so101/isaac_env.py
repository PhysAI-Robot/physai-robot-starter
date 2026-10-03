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
from typing import Any

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
from ...sim.studio import TABLE_RGBA
from ...sim.isaac.objects import (
    add_cube,
    add_static_box,
    add_target_pad,
)
from ...sim.isaac.scene import (
    add_ground_plane,
    add_studio_lighting,
    add_world_camera,
    checker_texture,
)
from ..base import RobotSpec, RobotTrainingContract
from ..description import RobotDescription, load_robot_description
from .contracts import (
    ALL_JOINT_NAMES,
    ARM_JOINT_NAMES,
    GRIPPER_JOINT_NAME,
    so101_training_contract,
)
from .kinematics import ArmKinematics
from .layout import DEFAULT_CUBE_X_RANGE, DEFAULT_CUBE_Y_RANGE, draw_cube_xy

REPO_ROOT = Path(__file__).resolve().parents[4]
_DESCRIPTION_PATH = Path(__file__).resolve().parent / "description.yaml"
HOME_QPOS = np.array([0.0, -1.05, 1.25, 0.75, 0.0], dtype=np.float64)
# Mean absolute pixel difference between a render with and without the robot
# below which the robot is taken to be missing. Measured on the front camera:
# a drawn robot gives ~10, two renders of the same state differ by ~1.3 (the
# denoiser), and a scene without the robot by ~0.3.
_ROBOT_RENDER_MIN_DIFF = 3.0


@dataclass
class GraspCubeConfig:
    """A minimal graspable cube for the tier-3 grasp-hold parity test
    (`ROADMAP.md`'s 2E) — not a `ManipulationSceneConfig` port: no table,
    target, layout, or randomization, just enough to grasp-and-hold. `position`
    and `friction` default to `SingleCubeFixedPlaceSceneConfig`'s own values
    (`cube_pos`, the sliding-friction component of `add_cube`'s
    `friction=[1.2, ...]`), placed directly on Isaac's ground plane rather
    than on a modeled table.
    """

    position: tuple[float, float, float] = (0.20, 0.08, 0.014)
    half_size: float = 0.014
    mass: float = 0.03
    friction: float = 1.2


@dataclass
class TableConfig:
    """The manipulation scene's table, a static collider. Defaults mirror
    `WorldSceneConfig.table_size`/`table_pos` and `build_manipulation_spec`'s
    table geom (colour, friction) exactly, so both simulators share one
    workspace. With a table, `target_pos` also gets MuJoCo's visual-only
    green `target_pad` disc.
    """

    position: tuple[float, float, float] = (0.30, 0.0, 0.01)
    half_extents: tuple[float, float, float] = (0.20, 0.25, 0.01)
    friction: float = 1.0
    rgba: tuple[float, float, float, float] = TABLE_RGBA
    target_radius: float = 0.035


@dataclass
class FrontCameraConfig:
    """A world-fixed camera, for `SO101VisualServoPolicy`'s default
    `camera="front"` (tier-4 parity, `ROADMAP.md`'s 2E) — not attached to
    the robot's articulation, unlike `IsaacEnvConfig.cameras`' other
    entries (link-mounted, via `RobotDescription.cameras`). Defaults mirror
    `WorldSceneConfig.front_cam_pos`/`front_cam_xyaxes` and
    `build_manipulation_spec`'s hardcoded `fovy=48` exactly, so both
    simulators frame the scene the same way.
    """

    position: tuple[float, float, float] = (0.62, 0.0, 0.38)
    x_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    y_axis: tuple[float, float, float] = (-0.45, 0.0, 0.9)
    fovy_deg: float = 48.0


def scene_objects(
    scene: Any,
) -> tuple[TableConfig, GraspCubeConfig, FrontCameraConfig, tuple[float, ...]]:
    """The table, cube, front camera and target a shared scene config describes.

    Takes the same `ManipulationSceneConfig` MuJoCo builds its scene from, so
    both simulators read one set of numbers. Friction, colour and the front
    camera's field of view are not scene fields (MuJoCo hardcodes them in
    `build_manipulation_spec`/`add_cube`); the dataclass defaults mirror those.
    """
    cube_names = tuple(getattr(scene, "cube_names", ("cube",)))
    if cube_names != ("cube",):
        raise ValueError(
            "SO101IsaacEnv supports single-cube scenes only; "
            f"this scene has cubes {cube_names}"
        )
    xyaxes = tuple(scene.front_cam_xyaxes)
    return (
        TableConfig(
            position=tuple(scene.table_pos),
            half_extents=tuple(scene.table_size),
            target_radius=scene.target_radius,
        ),
        GraspCubeConfig(
            position=tuple(scene.cube_pos),
            half_size=scene.cube_half,
            mass=scene.cube_mass,
        ),
        FrontCameraConfig(
            position=tuple(scene.front_cam_pos),
            x_axis=xyaxes[:3],
            y_axis=xyaxes[3:],
        ),
        tuple(scene.target_pos),
    )


@dataclass
class IsaacEnvConfig:
    """SO-101-on-Isaac simulation and observation settings.

    Narrower than `so101.mujoco_env.EnvConfig`: single-cube scenes only, a
    fixed target, and no domain randomization (Isaac's own randomization
    tooling is a separate integration). Give it the same `scene` MuJoCo uses
    and the table, cube, target, front camera and per-seed cube layout follow
    from it; without one, the low-level `cube`/`table`/`front_camera`/
    `target_pos` below describe a bare test scene.
    """

    scene: Any = (
        None  # a ManipulationSceneConfig (e.g. SingleCubeFixedPlaceSceneConfig)
    )
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
    cube: GraspCubeConfig | None = None
    front_camera: FrontCameraConfig | None = None
    table: TableConfig | None = None
    # Per-seed cube placement, as MuJoCo's `EnvConfig`: None means "randomize
    # when a scene is given" (the MuJoCo default), otherwise the cube stays at
    # its configured position.
    randomize_cube: bool | None = None
    cube_x_range: tuple[float, float] = DEFAULT_CUBE_X_RANGE
    cube_y_range: tuple[float, float] = DEFAULT_CUBE_Y_RANGE
    randomize_target: bool = False
    # (x, y) is what SO101VisualServoPolicy's TRANSFER/LOWER/RELEASE phases
    # actually read (see `_waypoint()`); z is recomputed from `rest_z`
    # separately, matching the MuJoCo side's own `target_pos` exactly.
    target_pos: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        parse_camera_resolution(self.camera_resolution)
        if self.randomize_target:
            raise ValueError("SO101IsaacEnv does not support randomize_target yet")
        if self.scene is not None:
            if self.scene.camera_resolution != self.camera_resolution:
                raise ValueError(
                    f"scene camera_resolution {self.scene.camera_resolution!r} "
                    f"differs from the env's {self.camera_resolution!r}"
                )
            table, cube, front, target = scene_objects(self.scene)
            self.table = self.table or table
            self.cube = self.cube or cube
            self.front_camera = self.front_camera or front
            if self.target_pos is None:
                self.target_pos = target
        if self.randomize_cube is None:
            self.randomize_cube = self.scene is not None
        self.cube_x_range = tuple(self.cube_x_range)
        self.cube_y_range = tuple(self.cube_y_range)
        # A session manifest's YAML `config:` arrives as plain dicts/lists.
        if isinstance(self.cube, dict):
            self.cube = GraspCubeConfig(**self.cube)
        if isinstance(self.table, dict):
            self.table = TableConfig(**self.table)
        if isinstance(self.front_camera, dict):
            self.front_camera = FrontCameraConfig(**self.front_camera)
        self.cameras = tuple(self.cameras)
        if self.target_pos is not None:
            self.target_pos = tuple(self.target_pos)


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

        if self.cfg.front_camera is not None:
            front = self.cfg.front_camera
            x_axis = np.asarray(front.x_axis, dtype=np.float64)
            y_axis = np.asarray(front.y_axis, dtype=np.float64)
            x_axis /= np.linalg.norm(x_axis)
            y_axis /= np.linalg.norm(y_axis)
            z_axis = np.cross(x_axis, y_axis)
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(
                quat, np.stack([x_axis, y_axis, z_axis], axis=1).reshape(9)
            )
            self._camera_prims["front"] = add_world_camera(
                self.stage,
                "/World_front_camera",
                position=front.position,
                quat_wxyz=tuple(quat),
                fovy_deg=front.fovy_deg,
                width=self._render_size[0],
                height=self._render_size[1],
            )

        if self.cfg.table is not None:
            table = self.cfg.table
            add_static_box(
                self.stage,
                "/World_table",
                position=table.position,
                half_extents=table.half_extents,
                friction=table.friction,
                rgba=table.rgba,
            )
            if self.cfg.target_pos is not None:
                add_target_pad(
                    self.stage,
                    "/World_target_pad",
                    position=self.cfg.target_pos,
                    radius=table.target_radius,
                )

        self._cube_path: str | None = None
        self._cube_body = None
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
        # Display/telemetry mirror for `physai.web.Host` (`--serve`): never
        # simulated, only refreshed from Isaac's joint state by `observe()`.
        self.model = self.kin.model
        self.data = mujoco.MjData(self.model)

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
        """
        mjcf_path = self.cfg.assets_root / self.description.mjcf
        model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        return ArmKinematics(model)

    @property
    def cube_pos(self) -> np.ndarray:
        if self._cube_path is None:
            raise AttributeError("this SO101IsaacEnv has no cube (cfg.cube is None)")
        positions, _ = self._cube_body.get_world_poses()
        return np.asarray(positions, dtype=np.float64)[0]

    @property
    def table_top(self) -> float:
        """Height of the surface objects rest on (the ground when no table)."""
        if self.cfg.table is None:
            return 0.0
        return self.cfg.table.position[2] + self.cfg.table.half_extents[2]

    @property
    def cube_half(self) -> float:
        if self.cfg.cube is None:
            raise AttributeError("this SO101IsaacEnv has no cube (cfg.cube is None)")
        return self.cfg.cube.half_size

    @property
    def rest_z(self) -> float:
        """Height a cube rests at — `SO101VisualServoPolicy`'s `_waypoint()`
        reads this instead of MuJoCo's `cfg.scene.table_pos`/`table_size`/
        `cube_half`: the table top plus the cube's half size when `cfg.table`
        is set, else wherever `cfg.cube` was placed (the ground plane)."""
        if self.cfg.cube is None:
            raise AttributeError("this SO101IsaacEnv has no cube (cfg.cube is None)")
        if self.cfg.table is not None:
            top = self.cfg.table.position[2] + self.cfg.table.half_extents[2]
            return top + self.cfg.cube.half_size
        return self.cfg.cube.position[2]

    @property
    def target_pos(self) -> np.ndarray:
        if self.cfg.target_pos is None:
            raise AttributeError(
                "this SO101IsaacEnv has no target (cfg.target_pos is None)"
            )
        return np.asarray(self.cfg.target_pos, dtype=np.float64)

    def camera_calibration(self, name: str) -> tuple[CameraIntrinsics, Pose]:
        """This camera's pinhole intrinsics and its pose in the base frame.

        `"front"` is world-fixed (`FrontCameraConfig`). `"wrist"` (any
        description camera) is link-mounted, so its pose is the parent
        link's current pose from the display mirror (`self.data`, refreshed
        by `observe()` before it asks) composed with the description's
        mount -- the same chain MuJoCo's own `cam_xmat` resolves. This is
        what lets `SO101VisualServoPolicy._refine_from_final_camera` re-aim
        the grasp from the wrist view during DESCEND; without it the grasp
        keeps the front camera's centimetre-level triangulation error.
        """
        if name == "front" and self.cfg.front_camera is not None:
            front = self.cfg.front_camera
            width, height = self.core.camera_size
            fovy_deg = front.fovy_deg
            x_axis = np.asarray(front.x_axis, dtype=np.float64)
            y_axis = np.asarray(front.y_axis, dtype=np.float64)
            x_axis /= np.linalg.norm(x_axis)
            y_axis /= np.linalg.norm(y_axis)
            z_axis = np.cross(x_axis, y_axis)
            # MuJoCo's own camera axes: x right, y up, z backward.
            rotation = np.stack([x_axis, y_axis, z_axis], axis=1)
            position = np.asarray(front.position, dtype=np.float64)
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
        self._reset_cube()
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

    def _reset_cube(self) -> None:
        """Put the cube at this episode's starting pose, at rest.

        `reset_simulation()` re-initializes physics but leaves a rigid body
        wherever the previous episode flung it, so a second episode would
        start with the cube off the table. The pose is the configured one, or
        with `randomize_cube` the same per-seed draw MuJoCo makes
        (`layout.draw_cube_xy` on `self.rng`) at the configured height.
        """
        if self._cube_path is None:
            return
        position = list(self.cfg.cube.position)
        if self.cfg.randomize_cube:
            position[:2] = draw_cube_xy(
                self.rng, self.cfg.cube_x_range, self.cfg.cube_y_range
            )
        self._cube_body.set_world_poses(
            positions=[position], orientations=[[1.0, 0.0, 0.0, 0.0]]
        )
        self._cube_body.set_velocities(
            linear_velocities=[[0.0, 0.0, 0.0]], angular_velocities=[[0.0, 0.0, 0.0]]
        )

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
    "FrontCameraConfig",
    "GraspCubeConfig",
    "IsaacEnvConfig",
    "SO101IsaacEnv",
    "TableConfig",
]
