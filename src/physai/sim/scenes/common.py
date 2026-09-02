"""Generic world and manipulation scene configuration primitives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ROBOT_XML = REPO_ROOT / "assets" / "so101" / "so101_new_calib.xml"
DEFAULT_STATIC_PAD_BODY = "gripper"
DEFAULT_MOVING_PAD_BODY = "moving_jaw_so101_v1"
DEFAULT_EE_SITE = "gripperframe"


@dataclass
class WorldSceneConfig:
    """Robot-independent world settings shared by scene implementations."""

    robot_xml: Path | None = None
    timestep: float = 0.002
    table_size: tuple[float, float, float] = (0.20, 0.25, 0.01)
    table_pos: tuple[float, float, float] = (0.30, 0.0, 0.01)
    target_pos: tuple[float, float, float] = (0.20, -0.10, 0.021)
    target_radius: float = 0.035
    camera_width: int = 320
    camera_height: int = 240
    front_cam_pos: tuple[float, float, float] = (0.62, 0.0, 0.38)
    front_cam_xyaxes: tuple[float, ...] = (0.0, 1.0, 0.0, -0.45, 0.0, 0.9)


@dataclass
class ManipulationSceneConfig(WorldSceneConfig):
    """World settings plus end-effector and gripper attachment details."""

    robot_xml: Path = DEFAULT_ROBOT_XML
    ee_site: str = DEFAULT_EE_SITE
    gripper_joint: str = "gripper"
    static_pad_body: str = DEFAULT_STATIC_PAD_BODY
    moving_pad_body: str = DEFAULT_MOVING_PAD_BODY
    pad_friction: tuple[float, float, float] = (2.0, 0.02, 0.001)
    pad_size: tuple[float, float, float] = (0.011, 0.009, 0.0015)
    pad_align_gripper_q: float = 0.25
    static_pad_pos: tuple[float, float, float] = (-0.0090, -0.0050, -0.0935)
    moving_pad_pos: tuple[float, float, float] = (-0.0117, -0.0700, 0.0228)
    # The wrist camera looks along -z of its own frame. With x = (-1, 0, 0) the
    # derived view direction pointed backwards and up, away from the workspace,
    # so this camera rendered a black frame for the whole episode. Negating the
    # x axis flips the view onto the jaws and the object below them while
    # keeping the original up vector, so the image is not also upside down.
    wrist_cam_pos: tuple[float, float, float] = (0.0, -0.07, 0.05)
    wrist_cam_xyaxes: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 0.7, 0.7)


# Compatibility name for callers from the original Phase 0 API. New code
# should choose WorldSceneConfig or ManipulationSceneConfig explicitly.
CommonSceneConfig = ManipulationSceneConfig


def _find_body(spec: mujoco.MjSpec, name: str):
    for body in spec.bodies:
        if body.name == name:
            return body
    raise KeyError(
        f"body {name!r} not found in {spec.modelname!r}. "
        f"Available: {[body.name for body in spec.bodies]}"
    )


def _pad_quats(cfg: ManipulationSceneConfig) -> dict[str, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(cfg.robot_xml))
    data = mujoco.MjData(model)
    grip_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, cfg.gripper_joint)
    data.qpos[model.jnt_qposadr[grip_jid]] = cfg.pad_align_gripper_q
    mujoco.mj_forward(model, data)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, cfg.ee_site)
    site_rotation = data.site_xmat[site_id].reshape(3, 3)
    quaternions: dict[str, np.ndarray] = {}
    for key, body_name in (("static", cfg.static_pad_body), ("moving", cfg.moving_pad_body)):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        local_rotation = data.xmat[body_id].reshape(3, 3).T @ site_rotation
        quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(quaternion, local_rotation.reshape(9))
        quaternions[key] = quaternion
    return quaternions


def build_manipulation_spec(cfg: ManipulationSceneConfig) -> mujoco.MjSpec:
    """Build a manipulation world with configurable robot attachments."""
    if cfg.robot_xml is None or not Path(cfg.robot_xml).exists():
        raise FileNotFoundError(
            f"{cfg.robot_xml} not found — run `python scripts/fetch_assets.py` first."
        )

    spec = mujoco.MjSpec.from_file(str(cfg.robot_xml))
    spec.option.timestep = cfg.timestep
    world = spec.worldbody

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
    world.add_light(
        pos=[0, 0, 2.0], dir=[0, 0, -1],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[0.7, 0.7, 0.7],
    )
    world.add_light(
        pos=[0.5, 0.5, 1.2], dir=[-0.4, -0.4, -1],
        type=mujoco.mjtLightType.mjLIGHT_SPOT,
        cutoff=60, exponent=10,
        diffuse=[0.3, 0.3, 0.3],
    )
    world.add_geom(
        name="physai_floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],
        pos=[0, 0, 0],
        material="physai_grid",
    )

    table = world.add_body(name="table", pos=list(cfg.table_pos))
    table.add_geom(
        name="table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(cfg.table_size),
        rgba=[0.75, 0.72, 0.66, 1.0],
        friction=[1.0, 0.005, 0.0001],
    )
    world.add_geom(
        name="target_pad",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[cfg.target_radius, 0.001, 0.0],
        pos=list(cfg.target_pos),
        rgba=[0.2, 0.7, 0.35, 0.55],
        contype=0,
        conaffinity=0,
    )
    world.add_site(
        name="target_site",
        pos=list(cfg.target_pos),
        size=[0.006, 0.006, 0.006],
        rgba=[0.2, 0.9, 0.4, 0.9],
    )

    quaternions = _pad_quats(cfg)

    def add_pad(body, position, quaternion, name: str) -> None:
        body.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=list(cfg.pad_size),
            pos=list(position),
            quat=list(quaternion),
            rgba=[0.12, 0.12, 0.14, 1.0],
            friction=list(cfg.pad_friction),
            condim=4,
            solimp=[0.95, 0.99, 0.001, 0.5, 2.0],
            solref=[0.004, 1.0],
            group=3,
        )

    add_pad(_find_body(spec, cfg.static_pad_body), cfg.static_pad_pos,
            quaternions["static"], "pad_static")
    add_pad(_find_body(spec, cfg.moving_pad_body), cfg.moving_pad_pos,
            quaternions["moving"], "pad_moving")

    world.add_camera(
        name="front",
        pos=list(cfg.front_cam_pos),
        xyaxes=list(cfg.front_cam_xyaxes),
        fovy=48,
    )
    _find_body(spec, cfg.static_pad_body).add_camera(
        name="wrist",
        pos=list(cfg.wrist_cam_pos),
        xyaxes=list(cfg.wrist_cam_xyaxes),
        fovy=62,
    )
    return spec


def build_common_spec(cfg: ManipulationSceneConfig) -> mujoco.MjSpec:
    """Compatibility alias for the original manipulation scene builder."""
    return build_manipulation_spec(cfg)


def add_cube(spec: mujoco.MjSpec, cfg: WorldSceneConfig, name: str,
             position, rgba, cube_half: float, cube_mass: float) -> None:
    cube = spec.worldbody.add_body(name=name, pos=list(position))
    cube.add_freejoint(name=f"{name}_free")
    cube.add_geom(
        name=f"{name}_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[cube_half] * 3,
        rgba=list(rgba),
        mass=cube_mass,
        friction=[1.2, 0.01, 0.0005],
        condim=4,
    )
    cube.add_site(
        name=f"{name}_site",
        pos=[0, 0, 0],
        size=[0.004] * 3,
        rgba=[1, 1, 0, 0.0],
    )
