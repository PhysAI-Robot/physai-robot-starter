"""Generic world and manipulation scene configuration primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]

# Matches the web viewer's Three.js scene background exactly
# (`scene.background = new THREE.Color(0xdfe6e2)` in
# src/physai/web/static/js/scene.js), so a MuJoCo render (the Tk viewer's
# free camera, a captured camera frame, an exported video) and the browser
# viewer show the same background color instead of MuJoCo's own default sky.
STUDIO_SKY_RGB: tuple[float, float, float] = (0.8745, 0.9020, 0.8863)

# Matches the web viewer's checker-textured floor exactly (`checkerTexture`
# in scene.js uses the same two hex colors, #e3e9e4 / #9fb0a8). A
# robot-mounted camera (e.g. so101's "front") often frames the floor rather
# than open sky, so the floor's own texture — not just the skybox — needs to
# be in the same palette for a MuJoCo render to look consistent with the
# browser viewer. The gap between the two tiles is wider than the sky/floor
# gap so the checker pattern stays legible without leaving the light palette.
STUDIO_FLOOR_RGB1: tuple[float, float, float] = (0.8902, 0.9137, 0.8941)
STUDIO_FLOOR_RGB2: tuple[float, float, float] = (0.6235, 0.6902, 0.6588)


def add_studio_sky(spec: mujoco.MjSpec) -> None:
    """Make the spec's skybox match the web viewer's background color.

    A robot's own MJCF (e.g. the upstream SO-101 `scene.xml`) may already
    define a skybox texture. A second `TEXTURE_SKYBOX` in the same model is
    not reliably overridden — different cameras in the same render can end
    up showing different skybox textures — so this overwrites the existing
    one in place instead of adding a second, and only adds a new one when
    the spec has none.
    """
    for texture in spec.textures:
        if texture.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            texture.builtin = mujoco.mjtBuiltin.mjBUILTIN_FLAT
            texture.rgb1 = list(STUDIO_SKY_RGB)
            texture.rgb2 = list(STUDIO_SKY_RGB)
            return
    spec.add_texture(
        name="physai_studio_sky",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
        rgb1=list(STUDIO_SKY_RGB),
        rgb2=list(STUDIO_SKY_RGB),
        width=256,
        height=256,
    )


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

    def to_metadata(self) -> dict:
        """This config as JSON-safe data for dataset metadata.

        Paths inside the repository are stored relative to it, so a dataset
        neither leaks the collecting machine's directory layout nor depends
        on where the repository was checked out. A path outside the
        repository has no portable form and is kept as given (forward-slash).
        """
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, Path):
                try:
                    value = value.relative_to(REPO_ROOT)
                except ValueError:
                    pass
                data[key] = value.as_posix()
        return data


@dataclass
class ManipulationSceneConfig(WorldSceneConfig):
    """World settings plus end-effector and gripper attachment details."""

    robot_xml: Path | None = None
    ee_site: str | None = None
    gripper_joint: str | None = None
    static_pad_body: str | None = None
    moving_pad_body: str | None = None
    wrist_body: str | None = None
    pad_friction: tuple[float, float, float] = (2.0, 0.02, 0.001)
    # MuJoCo models friction as a soft constraint, so a held object under a
    # constant load (a cube's own weight, ~0.3 N, against ~4 N of squeeze)
    # creeps out of the fingers at about 1 mm/s even though the friction
    # force is a small fraction of its limit; a cube held for ~15 s falls out
    # regardless of grip force. The no-slip post-solver removes that creep
    # (0.0 mm over 30 s) for ~30% more solver time. 0 restores the default.
    noslip_iterations: int = 5
    # The grasp pads are collision boxes standing in for the finger meshes (see
    # replace_jaw_collision). They are fitted to the SO-101 fingertips: each
    # pad's outer face is flush with the tip's inner face, its 12 x 12 mm
    # footprint is the bounding square of the tapered tip face and sits on the
    # finger centre line, and it is 6 mm thick so a squeezed cube cannot pass
    # through it into the finger. Only the last ~6 mm of each fingertip is a
    # flat face (behind it the lattice is recessed), so the pads are fitted to
    # that zone. `pad_align_gripper_q` (0.16 rad) is the gripper angle at
    # which the pad faces are one cube width (28 mm) apart at the pad
    # centres. Narrower pads (8-10 mm) fit the tip more tightly but make
    # `visual_servo` miss the cube on some seeds: it grasps up to ~15 mm off
    # the pinch centre.
    pad_size: tuple[float, float, float] = (0.006, 0.006, 0.003)
    replace_jaw_collision: bool = True
    pad_align_gripper_q: float = 0.16
    static_pad_pos: tuple[float, float, float] = (-0.0109, -0.0002, -0.0979)
    moving_pad_pos: tuple[float, float, float] = (-0.0093, -0.0753, 0.0190)
    # Rotation of the moving pad about its lateral axis, on top of being
    # parallel to the static pad at `pad_align_gripper_q`. The moving finger's
    # face is not parallel to the static one (the fingers form a V, about 8
    # degrees apart here), so this lays the pad along that face. Positive
    # raises the face toward the tip.
    moving_pad_tilt: float = 0.1348
    # The pads render in the web viewer (a group-3 box drawn by its rgba) so
    # their fit can be checked by eye; MuJoCo camera renders skip group 3, so
    # dataset images are unaffected. Set the alpha (last value) to 0 to hide.
    pad_rgba: tuple[float, float, float, float] = (0.95, 0.6, 0.1, 0.6)
    # The wrist camera looks along -z of its own frame. With x = (-1, 0, 0) the
    # derived view direction pointed backwards and up, away from the workspace,
    # so this camera rendered a black frame for the whole episode. Negating the
    # x axis flips the view onto the jaws and the object below them while
    # keeping the original up vector, so the image is not also upside down.
    wrist_cam_pos: tuple[float, float, float] = (0.0, -0.07, 0.05)
    wrist_cam_xyaxes: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 0.7, 0.7)
    clutter_count: int = 0
    clutter_size: tuple[float, float, float] = (0.018, 0.018, 0.025)

    def build_spec(self) -> mujoco.MjSpec:
        """Build the MuJoCo spec for this scene, objects included."""
        return build_manipulation_spec(self)

    def build_model(self) -> tuple[mujoco.MjModel, mujoco.MjSpec]:
        spec = self.build_spec()
        return spec.compile(), spec


def export_xml(path: Path, cfg: ManipulationSceneConfig) -> Path:
    """Write a scene to disk as MJCF."""
    spec = cfg.build_spec()
    spec.compile()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(spec.to_xml(), encoding="utf-8")
    return path


def _find_body(spec: mujoco.MjSpec, name: str):
    for body in spec.bodies:
        if body.name == name:
            return body
    raise KeyError(
        f"body {name!r} not found in {spec.modelname!r}. "
        f"Available: {[body.name for body in spec.bodies]}"
    )


def _validate_robot_attachment(cfg: ManipulationSceneConfig) -> None:
    missing = [
        name
        for name in (
            "robot_xml",
            "ee_site",
            "gripper_joint",
            "static_pad_body",
            "moving_pad_body",
        )
        if getattr(cfg, name) is None
    ]
    if missing:
        raise ValueError(
            "scene requires robot attachment configuration: " + ", ".join(missing)
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
    for key, body_name in (
        ("static", cfg.static_pad_body),
        ("moving", cfg.moving_pad_body),
    ):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        local_rotation = data.xmat[body_id].reshape(3, 3).T @ site_rotation
        if key == "moving":
            c, s = np.cos(cfg.moving_pad_tilt), np.sin(cfg.moving_pad_tilt)
            local_rotation = local_rotation @ np.array(
                [[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]]
            )
        quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(quaternion, local_rotation.reshape(9))
        quaternions[key] = quaternion
    return quaternions


def _replace_jaw_collision(spec: mujoco.MjSpec, cfg: ManipulationSceneConfig) -> None:
    if not cfg.replace_jaw_collision:
        return
    for body_name in (cfg.static_pad_body, cfg.moving_pad_body):
        body = _find_body(spec, body_name)
        for geom in body.geoms:
            if geom.type == mujoco.mjtGeom.mjGEOM_MESH and geom.group == 3:
                geom.contype = 0
                geom.conaffinity = 0


def _add_wrist_jog_site(spec: mujoco.MjSpec, cfg: ManipulationSceneConfig) -> None:
    """Add a zero-offset "wristframe" site so a restricted-joint jog IK can
    target the wrist rather than the gripper tip. Added here (not hand-edited
    into the fetched MJCF) because `assets/` is downloaded by
    `scripts/fetch_assets.py` and not committed to the repo.
    """
    if cfg.wrist_body is None:
        return
    _find_body(spec, cfg.wrist_body).add_site(name="wristframe", pos=[0, 0, 0])


def build_manipulation_spec(cfg: ManipulationSceneConfig) -> mujoco.MjSpec:
    _validate_robot_attachment(cfg)
    """Build a manipulation world with configurable robot attachments."""
    if cfg.robot_xml is None or not Path(cfg.robot_xml).exists():
        raise FileNotFoundError(
            f"{cfg.robot_xml} not found — run `python scripts/fetch_assets.py` first."
        )

    spec = mujoco.MjSpec.from_file(str(cfg.robot_xml))
    spec.option.timestep = cfg.timestep
    spec.option.noslip_iterations = cfg.noslip_iterations
    _replace_jaw_collision(spec, cfg)
    _add_wrist_jog_site(spec, cfg)
    world = spec.worldbody

    add_studio_sky(spec)
    spec.add_texture(
        name="physai_grid",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=list(STUDIO_FLOOR_RGB1),
        rgb2=list(STUDIO_FLOOR_RGB2),
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
        pos=[0, 0, 2.0],
        dir=[0, 0, -1],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[0.7, 0.7, 0.7],
    )
    world.add_light(
        pos=[0.5, 0.5, 1.2],
        dir=[-0.4, -0.4, -1],
        type=mujoco.mjtLightType.mjLIGHT_SPOT,
        cutoff=60,
        exponent=10,
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
    if cfg.clutter_count < 0:
        raise ValueError("clutter_count must be non-negative")
    for index in range(cfg.clutter_count):
        world.add_geom(
            name=f"physai_clutter_{index}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.14 + 0.03 * index, -0.16 + 0.06 * index, cfg.clutter_size[2]],
            size=list(cfg.clutter_size),
            rgba=[0.35, 0.42, 0.48, 1.0],
            friction=[0.8, 0.01, 0.0001],
        )

    quaternions = _pad_quats(cfg)

    def add_pad(body, position, quaternion, name: str) -> None:
        body.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=list(cfg.pad_size),
            pos=list(position),
            quat=list(quaternion),
            # A collision-only proxy for the jaw mesh (see
            # _replace_jaw_collision). Its rgba only affects the web viewer
            # (scene.js sets transparent/opacity from rgba[3]); alpha 0 hides
            # it without touching contact behavior. Its solref time constant
            # is already at the stability floor (2 x timestep): a stiffer
            # direct solref throws the cube out of the grasp or blows up the
            # arm at this timestep. The high impedance (solimp) is the safe
            # remaining lever.
            rgba=list(cfg.pad_rgba),
            friction=list(cfg.pad_friction),
            condim=4,
            solimp=[0.99, 0.999, 0.001, 0.5, 2.0],
            solref=[0.004, 1.0],
            group=3,
        )

    add_pad(
        _find_body(spec, cfg.static_pad_body),
        cfg.static_pad_pos,
        quaternions["static"],
        "pad_static",
    )
    add_pad(
        _find_body(spec, cfg.moving_pad_body),
        cfg.moving_pad_pos,
        quaternions["moving"],
        "pad_moving",
    )

    world.add_camera(
        name="front",
        pos=list(cfg.front_cam_pos),
        xyaxes=list(cfg.front_cam_xyaxes),
        fovy=48,
    )
    camera_body = next(
        (body for body in spec.bodies if body.name == "wrist_camera"), None
    )
    if camera_body is not None:
        # The official camera variant contains the calibrated mount and camera
        # body. The sensor is added here because the upstream model only
        # describes the physical camera mesh, not a MuJoCo render sensor.
        camera_body.add_camera(
            name="wrist",
            # Keep the virtual optical centre just outside the physical lens
            # housing; placing it at the body origin makes the housing occlude
            # the rendered image as a large black spot.
            pos=[0.0, 0.0, 0.025],
            # The official camera body uses +z as its optical direction while
            # MuJoCo camera sensors look along local -z.
            xyaxes=[1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
            fovy=62,
        )
    else:
        _find_body(spec, cfg.static_pad_body).add_camera(
            name="wrist",
            pos=list(cfg.wrist_cam_pos),
            xyaxes=list(cfg.wrist_cam_xyaxes),
            fovy=62,
        )
    return spec


def add_cube(
    spec: mujoco.MjSpec,
    cfg: WorldSceneConfig,
    name: str,
    position,
    rgba,
    cube_half: float,
    cube_mass: float,
) -> None:
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
