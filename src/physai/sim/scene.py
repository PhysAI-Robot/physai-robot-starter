"""Scene composition for the SO-101 pick-and-place task.

The upstream MJCF (assets/so101/so101_new_calib.xml) is the bare arm. Rather
than maintaining a forked XML, we load it with MjSpec and add the task world
programmatically: floor, table, cube, target pad, cameras, and rubber finger
pads on the two gripper links.

Why finger pads: the stock gripper collision geoms are STL meshes with default
friction. Mesh-on-mesh grasping in MuJoCo is slippery and jitters. Two thin
high-friction boxes on `gripper` and `moving_jaw_so101_v1` make grasps stable
without touching the upstream description. Tune in configs/sim.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROBOT_XML = REPO_ROOT / "assets" / "so101" / "so101_new_calib.xml"

# Bodies the pads attach to (verified against so101_new_calib.xml).
STATIC_FINGER_BODY = "gripper"
MOVING_FINGER_BODY = "moving_jaw_so101_v1"
EE_SITE = "gripperframe"


@dataclass
class SceneConfig:
    robot_xml: Path = DEFAULT_ROBOT_XML
    timestep: float = 0.002

    # The table must start clear of the robot base at the origin — an
    # overlapping slab silently jams shoulder_pan against its force limit.
    table_size: tuple[float, float, float] = (0.20, 0.25, 0.01)
    table_pos: tuple[float, float, float] = (0.30, 0.0, 0.01)

    cube_half: float = 0.014
    cube_pos: tuple[float, float, float] = (0.20, 0.08, 0.036)
    cube_mass: float = 0.03
    cube_rgba: tuple[float, float, float, float] = (0.85, 0.25, 0.2, 1.0)

    # num_cubes=1 (default) reproduces the original single-cube scene exactly
    # — every existing test and dataset assumes that body is named "cube".
    # num_cubes=3 swaps it for three independently-colored cubes (sorting
    # task) named cube_red/cube_blue/cube_yellow instead; there is no
    # "cube" body in that mode. Colors deliberately exclude green — the
    # target pad is green, and "put the green cube on the green pad" reads
    # as a bug report waiting to happen.
    num_cubes: int = 1
    sorting_cube_names: tuple[str, ...] = ("cube_red", "cube_blue", "cube_yellow")
    sorting_cube_rgba: tuple[tuple[float, float, float, float], ...] = (
        (0.85, 0.25, 0.2, 1.0),
        (0.2, 0.35, 0.85, 1.0),
        (0.9, 0.8, 0.15, 1.0),
    )

    target_pos: tuple[float, float, float] = (0.20, -0.10, 0.021)
    target_radius: float = 0.035

    # Contact pads: flat plates on the two jaw faces. Positions were measured
    # off the jaw meshes of so101_new_calib.xml (nearest distal vertex pair,
    # gripper joint at 0). Orientation is computed at build time from the EE
    # site frame — a sphere or a mis-rotated box squeezes a cube out sideways
    # instead of holding it. Retune the positions if you swap to old_calib.
    pad_friction: tuple[float, float, float] = (2.0, 0.02, 0.001)
    # Half-extents in the *site* frame: (along approach, lateral, thickness).
    pad_size: tuple[float, float, float] = (0.011, 0.009, 0.0015)
    pad_align_gripper_q: float = 0.25   # jaw angle the faces are made parallel at
    static_pad_pos: tuple[float, float, float] = (-0.0090, -0.0050, -0.0935)
    moving_pad_pos: tuple[float, float, float] = (-0.0117, -0.0700, 0.0228)

    camera_width: int = 320
    camera_height: int = 240
    front_cam_pos: tuple[float, float, float] = (0.62, 0.0, 0.38)
    front_cam_xyaxes: tuple[float, ...] = (0.0, 1.0, 0.0, -0.45, 0.0, 0.9)
    wrist_cam_pos: tuple[float, float, float] = (0.0, -0.05, 0.03)
    wrist_cam_xyaxes: tuple[float, ...] = (-1.0, 0.0, 0.0, 0.0, 0.7, 0.7)


def _find_body(spec: mujoco.MjSpec, name: str):
    for b in spec.bodies:
        if b.name == name:
            return b
    raise KeyError(
        f"body {name!r} not found in {spec.modelname!r}. "
        f"Available: {[b.name for b in spec.bodies]}"
    )


def _pad_quats(cfg: SceneConfig) -> dict[str, np.ndarray]:
    """Body-local quaternions that align each pad's faces with the jaw gap.

    Compiles the bare arm once, poses the jaw at `pad_align_gripper_q`, and
    expresses the EE site's world orientation in each finger body's frame. The
    pad boxes then have their thin axis along the pinch direction on both jaws,
    which is what makes the grasp hold instead of ejecting the object.
    """
    model = mujoco.MjModel.from_xml_path(str(cfg.robot_xml))
    data = mujoco.MjData(model)
    grip_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
    data.qpos[model.jnt_qposadr[grip_jid]] = cfg.pad_align_gripper_q
    mujoco.mj_forward(model, data)

    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    R_site = data.site_xmat[sid].reshape(3, 3)

    out: dict[str, np.ndarray] = {}
    for key, body in (("static", STATIC_FINGER_BODY), ("moving", MOVING_FINGER_BODY)):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        R_local = data.xmat[bid].reshape(3, 3).T @ R_site
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, R_local.reshape(9))
        out[key] = quat
    return out


def build_spec(cfg: SceneConfig | None = None) -> mujoco.MjSpec:
    """Load the SO-101 and add the pick-and-place world. Returns an MjSpec."""
    cfg = cfg or SceneConfig()
    if not Path(cfg.robot_xml).exists():
        raise FileNotFoundError(
            f"{cfg.robot_xml} not found — run `python scripts/fetch_assets.py` first."
        )

    spec = mujoco.MjSpec.from_file(str(cfg.robot_xml))
    spec.option.timestep = cfg.timestep

    # --- world dressing -------------------------------------------------
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
    world = spec.worldbody
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

    # --- table ----------------------------------------------------------
    table = world.add_body(name="table", pos=list(cfg.table_pos))
    table.add_geom(
        name="table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(cfg.table_size),
        rgba=[0.75, 0.72, 0.66, 1.0],
        friction=[1.0, 0.005, 0.0001],
    )

    # --- target pad (visual only, no contact) ---------------------------
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

    # --- cube(s) (the manipuland) ----------------------------------------
    def _add_cube(name: str, pos, rgba) -> None:
        cube = world.add_body(name=name, pos=list(pos))
        cube.add_freejoint(name=f"{name}_free")
        cube.add_geom(
            name=f"{name}_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[cfg.cube_half] * 3,
            rgba=list(rgba),
            mass=cfg.cube_mass,
            friction=[1.2, 0.01, 0.0005],
            condim=4,
        )
        cube.add_site(name=f"{name}_site", pos=[0, 0, 0], size=[0.004] * 3,
                      rgba=[1, 1, 0, 0.0])

    if cfg.num_cubes == 1:
        _add_cube("cube", cfg.cube_pos, cfg.cube_rgba)
    elif cfg.num_cubes == len(cfg.sorting_cube_names):
        # Initial positions here don't matter — env.reset() overwrites them
        # every episode. Offset them apart only so a pre-reset compile/render
        # (e.g. export_xml) doesn't show three cubes stacked on one spot.
        for i, (name, rgba) in enumerate(zip(cfg.sorting_cube_names, cfg.sorting_cube_rgba)):
            pos = (cfg.cube_pos[0], cfg.cube_pos[1] + 0.06 * i, cfg.cube_pos[2])
            _add_cube(name, pos, rgba)
    else:
        raise ValueError(
            f"num_cubes={cfg.num_cubes} not supported — use 1 (default) or "
            f"{len(cfg.sorting_cube_names)} (sorting_cube_names)"
        )

    # --- flat friction pads on both jaw faces ---------------------------
    quats = _pad_quats(cfg)

    def _add_pad(body, pos, quat, name: str) -> None:
        body.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=list(cfg.pad_size),
            pos=list(pos),
            quat=list(quat),
            rgba=[0.12, 0.12, 0.14, 1.0],
            friction=list(cfg.pad_friction),
            condim=4,
            solimp=[0.95, 0.99, 0.001, 0.5, 2.0],
            solref=[0.004, 1.0],
            group=3,
        )

    _add_pad(_find_body(spec, STATIC_FINGER_BODY), cfg.static_pad_pos,
             quats["static"], "pad_static")
    _add_pad(_find_body(spec, MOVING_FINGER_BODY), cfg.moving_pad_pos,
             quats["moving"], "pad_moving")

    # --- cameras --------------------------------------------------------
    world.add_camera(
        name="front",
        pos=list(cfg.front_cam_pos),
        xyaxes=list(cfg.front_cam_xyaxes),
        fovy=48,
    )
    wrist_body = _find_body(spec, STATIC_FINGER_BODY)
    wrist_body.add_camera(
        name="wrist",
        pos=list(cfg.wrist_cam_pos),
        xyaxes=list(cfg.wrist_cam_xyaxes),
        fovy=62,
    )

    return spec


def build_model(cfg: SceneConfig | None = None) -> tuple[mujoco.MjModel, mujoco.MjSpec]:
    spec = build_spec(cfg)
    return spec.compile(), spec


def export_xml(path: Path, cfg: SceneConfig | None = None) -> Path:
    """Write the composed scene to disk — handy for `python -m mujoco.viewer`."""
    spec = build_spec(cfg)
    spec.compile()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(spec.to_xml(), encoding="utf-8")
    return path


if __name__ == "__main__":
    model, _ = build_model()
    print(f"compiled: nq={model.nq} nu={model.nu} nbody={model.nbody} ncam={model.ncam}")
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
             for i in range(model.ncam)]
    print("cameras:", names)
