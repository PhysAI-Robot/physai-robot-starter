"""Generic world and manipulation scene configuration primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar

import mujoco

from ....robots.description import RobotDescription
from ...studio import STUDIO_FLOOR_RGB1, STUDIO_FLOOR_RGB2, STUDIO_SKY_RGB, TABLE_RGBA
from ...workspace import (
    CUBE_FRICTION,
    FRONT_CAMERA_FOVY_DEG,
    TABLE_FRICTION,
    TARGET_RGBA,
    CubeSpec,
    WorkspaceConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[5]


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
class WorldSceneConfig(WorkspaceConfig):
    """The workspace plus the MuJoCo-only settings of one scene."""

    robot_xml: Path | None = None
    timestep: float = 0.002

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

    # How a robot should place this scene's objects each episode (a name the
    # robot maps to its own layout). Not a field: it is not run configuration.
    layout_kind: ClassVar[str | None] = None

    robot_xml: Path | None = None
    # The robot's sim-neutral frames, cameras, contact pads, and MuJoCo-only
    # tuning (see `robots.description.RobotDescription`); supplied by the
    # robot's scene defaults (`robots/so101/scene.py`), which have no generic
    # value for an unattached scene.
    description: RobotDescription | None = None
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
        name for name in ("robot_xml", "description") if getattr(cfg, name) is None
    ]
    if missing:
        raise ValueError(
            "scene requires robot attachment configuration: " + ", ".join(missing)
        )


def _find_body_optional(spec: mujoco.MjSpec, name: str):
    for body in spec.bodies:
        if body.name == name:
            return body
    return None


def apply_description(
    spec: mujoco.MjSpec, desc: RobotDescription, *, include_contact_pads: bool = True
) -> None:
    """Add a robot's sim-neutral frames and cameras to its compiled spec, plus
    (when `include_contact_pads`) its contact pads and MuJoCo-only tuning
    from `desc.mujoco_overrides()`.

    A frame or camera whose `parent_link` is absent from this spec is skipped
    rather than failing: `desc.cameras` may describe mounts for more than one
    upstream MJCF variant (see `robots/so101/description.yaml`), and only one
    variant is loaded at a time.

    `include_contact_pads=False` is for a shared multi-robot world (see
    `robots/so101/shared.py`), which attaches a robot for kinematics and
    control but not grasp-pad physics; every other consumer wants pads too.
    """
    for frame in desc.frames:
        body = _find_body_optional(spec, frame.parent_link)
        if body is not None:
            body.add_site(name=frame.name, pos=list(frame.pos), quat=list(frame.quat))

    mounted_cameras: set[str] = set()
    for camera in desc.cameras:
        # Two entries may name the same logical camera with different mounts
        # for different upstream MJCF variants (see `description.yaml`'s two
        # "wrist" entries); take the first whose parent link this spec has,
        # in the description's own priority order, and skip the rest.
        if camera.name in mounted_cameras:
            continue
        body = _find_body_optional(spec, camera.parent_link)
        if body is not None:
            body.add_camera(
                name=camera.name,
                pos=list(camera.pos),
                quat=list(camera.quat),
                fovy=camera.fovy_deg,
            )
            mounted_cameras.add(camera.name)

    if not include_contact_pads:
        return

    overrides = desc.mujoco_overrides()
    spec.option.noslip_iterations = overrides.get("noslip_iterations", 0)
    pad_style = overrides.get("pad", {})
    for pad in desc.contact_pads:
        body = _find_body(spec, pad.parent_link)
        if pad.disable_parent_mesh_collision:
            for geom in body.geoms:
                if geom.type == mujoco.mjtGeom.mjGEOM_MESH and geom.group == 3:
                    geom.contype = 0
                    geom.conaffinity = 0
        body.add_geom(
            name=pad.name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=list(pad.half_size),
            pos=list(pad.pos),
            quat=list(pad.quat),
            # A collision-only proxy for the jaw mesh. Its rgba only affects
            # the web viewer (scene.js sets transparent/opacity from
            # rgba[3]); alpha 0 hides it without touching contact behavior.
            rgba=list(pad_style.get("rgba", (1.0, 1.0, 1.0, 1.0))),
            friction=list(pad.friction),
            condim=int(pad_style.get("condim", 3)),
            solimp=list(pad_style.get("solimp", (0.9, 0.95, 0.001, 0.5, 2.0))),
            solref=list(pad_style.get("solref", (0.02, 1.0))),
            group=int(pad_style.get("group", 0)),
        )


def build_manipulation_spec(cfg: ManipulationSceneConfig) -> mujoco.MjSpec:
    """Build a manipulation world with configurable robot attachments."""
    _validate_robot_attachment(cfg)
    if cfg.robot_xml is None or not Path(cfg.robot_xml).exists():
        raise FileNotFoundError(
            f"{cfg.robot_xml} not found — run `python scripts/fetch_assets.py` first."
        )

    spec = mujoco.MjSpec.from_file(str(cfg.robot_xml))
    spec.option.timestep = cfg.timestep
    # The offscreen framebuffer must hold the chosen camera image (MuJoCo's
    # default is 640 x 480, too small for 1280 x 720).
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, cfg.camera_width)
    spec.visual.global_.offheight = max(
        spec.visual.global_.offheight, cfg.camera_height
    )
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
        rgba=list(TABLE_RGBA),
        friction=list(TABLE_FRICTION),
    )
    world.add_geom(
        name="target_pad",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[cfg.target_radius, 0.001, 0.0],
        pos=list(cfg.target_pos),
        rgba=list(TARGET_RGBA),
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

    world.add_camera(
        name="front",
        pos=list(cfg.front_cam_pos),
        xyaxes=list(cfg.front_cam_xyaxes),
        fovy=FRONT_CAMERA_FOVY_DEG,
    )
    for cube in cfg.cubes():
        add_cube(spec, cube)
    apply_description(spec, cfg.description)
    return spec


def add_cube(spec: mujoco.MjSpec, cube: CubeSpec) -> None:
    body = spec.worldbody.add_body(name=cube.name, pos=list(cube.position))
    body.add_freejoint(name=f"{cube.name}_free")
    body.add_geom(
        name=f"{cube.name}_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[cube.half_size] * 3,
        rgba=list(cube.rgba),
        mass=cube.mass,
        friction=list(CUBE_FRICTION),
        condim=4,
    )
    body.add_site(
        name=f"{cube.name}_site",
        pos=[0, 0, 0],
        size=[0.004] * 3,
        rgba=[1, 1, 0, 0.0],
    )
