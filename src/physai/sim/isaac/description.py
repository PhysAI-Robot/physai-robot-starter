"""Apply a `RobotDescription` (`physai.robots.description`) to an Isaac Sim
USD stage: import the URDF, select a physics variant, and layer the
robot's own frames, cameras, contact-pad friction, and actuator gains on
top — the Isaac analogue of `sim.mujoco.scenes.common.apply_description`.

Findings this module encodes, verified against Isaac Sim 6.1.0.0 on an
RTX 3060 (see the plan's Phase 2.0/2.2 notes; there is no public spec for
most of this, so treat the version pin as load-bearing):

- `isaacsim.asset.importer.urdf.{URDFImporter, URDFImporterConfig}` is the
  stable Python import API (`omni.kit.commands`' "URDFParseAndImportFile"
  does not exist in 6.1).
- The imported robot's physics representation is a USD variant set
  (`Physics`: "mujoco" | "physx" | "physics" | "none") with no default
  selection — traversal finds zero joints until one is selected.
- `isaacsim.core.experimental.prims.Articulation` is the current, supported
  joint-control API; `isaacsim.core.prims`/`isaacsim.core.api` (including
  `World`) are deprecated in 6.1 (they live under an `extsdeprecated`
  install path and fail once physics has not been separately initialized).
- `Articulation.set_dof_gains(stiffnesses, dampings)` and
  `.set_dof_max_efforts(max_efforts)` take **radians**, matching MuJoCo's
  and URDF's own convention directly — our SO-101 MJCF gains
  (stiffness=998.22, damping=2.731, force_limit=3.35) transferred with no
  unit conversion and settled a 0.3 rad step to within 2e-4 rad in 3 s.
- The URDF importer suffixes every link's Xform prim with "_link"
  (`"gripper"` -> `"gripper_link"`); `RobotDescription`'s `parent_link`
  values are MJCF body names, so this module normalizes them for the USD
  side instead of the description needing two names per part.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...robots.description import RobotDescription

DEFAULT_PHYSICS_VARIANT = "physx"


def urdf_link_name(mjcf_body_name: str) -> str:
    """The URDF-imported Xform prim name for an MJCF body name.

    See this module's docstring: empirically verified against the SO-101
    import, not derived from a documented naming guarantee. A future robot
    whose importer output differs needs this updated, or overridden per
    robot rather than assumed generic.
    """
    return f"{mjcf_body_name}_link"


def import_robot(
    desc: RobotDescription,
    assets_root: Path,
    *,
    usd_out_dir: Path,
    variant: str = DEFAULT_PHYSICS_VARIANT,
    fix_base: bool = True,
) -> tuple[str, str]:
    """Import `desc.urdf`, select a physics variant, and return
    `(usd_path, robot_prim_path)`. The stage is left open on the process's
    `omni.usd` context (`get_stage()` below returns it) — callers must not
    also open it via a bare `Usd.Stage.Open()`, which raced against this
    context's own edits and raised a USD "concurrent changes" error when
    tried during development.

    `usd_out_dir` is a directory the importer may write many files under
    (a `<urdf-stem>/` subtree, not one file); callers own its lifetime the
    way `scripts/fetch_assets.py` owns `assets/`, since Isaac's own output
    is not something this project fetches or commits either.
    """
    import omni.usd
    from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig

    urdf_path = Path(assets_root) / desc.urdf
    if not urdf_path.exists():
        raise FileNotFoundError(
            f"{urdf_path} not found — run `python scripts/fetch_assets.py` first."
        )

    config = URDFImporterConfig()
    config.urdf_path = str(urdf_path)
    config.usd_path = str(usd_out_dir)
    config.fix_base = fix_base
    config.merge_fixed_joints = False
    config.collision_from_visuals = False

    importer = URDFImporter(config)
    usd_path = importer.import_urdf()

    ctx = omni.usd.get_context()
    ctx.open_stage(usd_path)
    stage = ctx.get_stage()
    root = stage.GetDefaultPrim()
    ok = root.GetVariantSets().GetVariantSet("Physics").SetVariantSelection(variant)
    if not ok:
        raise ValueError(f"USD Physics variant {variant!r} not available in {usd_path}")
    stage.GetRootLayer().Save()
    return usd_path, f"/{root.GetName()}"


def get_stage() -> Any:
    """The USD stage `import_robot()` left open on the omni context."""
    import omni.usd

    return omni.usd.get_context().get_stage()


def apply_actuators(articulation: Any, desc: RobotDescription) -> None:
    """Set joint drive gains, effort limits, and control-range limits from
    `desc.actuators` onto an already-constructed experimental `Articulation`.

    Every joint keeps whatever the importer gave it (a generic default) if
    it has no matching entry in `desc.actuators`, rather than being zeroed:
    a partial description should not disable joints it does not mention.
    """
    import numpy as np

    dof_names = list(articulation.dof_names)
    stiffness, damping = articulation.get_dof_gains()
    stiffness, damping = np.asarray(stiffness).copy(), np.asarray(damping).copy()
    max_effort = np.asarray(articulation.get_dof_max_efforts()).copy()
    lower, upper = articulation.get_dof_limits()
    lower, upper = np.asarray(lower).copy(), np.asarray(upper).copy()

    missing = [a.joint for a in desc.actuators if a.joint not in dof_names]
    if missing:
        raise KeyError(
            f"description actuators name joints absent from the articulation: "
            f"{missing}; available: {dof_names}"
        )

    for actuator in desc.actuators:
        index = dof_names.index(actuator.joint)
        stiffness[0, index] = actuator.stiffness
        damping[0, index] = actuator.damping
        max_effort[0, index] = actuator.force_limit
        lower[0, index], upper[0, index] = actuator.ctrl_range

    articulation.set_dof_gains(stiffness, damping)
    articulation.set_dof_max_efforts(max_effort)
    articulation.set_dof_limits(lower, upper)


def apply_frames(stage: Any, desc: RobotDescription) -> None:
    """Add `desc.frames` as zero-extent Xforms under their URDF link prim
    (e.g. `wristframe`), the Isaac analogue of MuJoCo's added sites.
    """
    from pxr import Gf, UsdGeom

    for frame in desc.frames:
        parent_path = f"/{stage.GetDefaultPrim().GetName()}/Geometry/{_geometry_subpath(stage, frame.parent_link)}"
        parent = stage.GetPrimAtPath(parent_path)
        if not parent.IsValid():
            continue
        xform = UsdGeom.Xform.Define(stage, parent_path + "/" + frame.name)
        xform.AddTranslateOp().Set(Gf.Vec3d(*frame.pos))
        w, x, y, z = frame.quat
        xform.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


#: USD's own default camera vertical aperture (mm), used as the fixed point
#: the MuJoCo-style `fovy` (full vertical angle) is solved against.
DEFAULT_VERTICAL_APERTURE_MM = 20.955


def fovy_to_focal_length(
    fovy_deg: float, *, vertical_aperture_mm: float = DEFAULT_VERTICAL_APERTURE_MM
) -> float:
    """A USD camera's focal length (mm) for a MuJoCo-style vertical FOV.

    Solves the pinhole vertical-FOV relation
    (`vertical_aperture = 2 * focal_length * tan(fovy / 2)`) for focal
    length, matching `fovy` the way MuJoCo itself defines it: the full
    vertical angle. Not independently re-verified against a rendered
    projection (see this module's docstring) — Phase 3's camera parity tier
    is what validates or corrects this.
    """
    import math

    if not 0.0 < fovy_deg < 180.0:
        raise ValueError(f"fovy_deg must be in (0, 180), got {fovy_deg!r}")
    return vertical_aperture_mm / (2.0 * math.tan(math.radians(fovy_deg) / 2.0))


def apply_cameras(stage: Any, desc: RobotDescription) -> dict[str, str]:
    """Add `desc.cameras` as USD cameras under their URDF link prim.

    Returns `{camera_name: prim_path}` for the cameras actually added; a
    camera whose `parent_link` this stage lacks is skipped (see
    `sim.mujoco.scenes.common.apply_description`'s matching MuJoCo behavior for
    the same reason — one description may cover more than one upstream
    variant). Only the first match per camera name is added, in the
    description's own priority order.
    """
    from pxr import Gf, UsdGeom

    added: dict[str, str] = {}
    for camera in desc.cameras:
        if camera.name in added:
            continue
        parent_subpath = _geometry_subpath(stage, camera.parent_link)
        if parent_subpath is None:
            continue
        parent_path = f"/{stage.GetDefaultPrim().GetName()}/Geometry/{parent_subpath}"
        prim_path = f"{parent_path}/{camera.name}_camera"
        cam = UsdGeom.Camera.Define(stage, prim_path)
        cam.AddTranslateOp().Set(Gf.Vec3d(*camera.pos))
        w, x, y, z = camera.quat
        cam.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
        vertical_aperture_mm = DEFAULT_VERTICAL_APERTURE_MM
        cam.CreateFocalLengthAttr(
            fovy_to_focal_length(
                camera.fovy_deg, vertical_aperture_mm=vertical_aperture_mm
            )
        )
        cam.CreateVerticalApertureAttr(vertical_aperture_mm)
        cam.CreateHorizontalApertureAttr(
            vertical_aperture_mm * camera.width / camera.height
        )
        added[camera.name] = prim_path
    return added


def apply_contact_friction(stage: Any, desc: RobotDescription) -> None:
    """Set a PhysX material's static/dynamic friction from each contact
    pad's `friction[0]` (MuJoCo's sliding-friction component; the only one
    a PhysX isotropic material has a direct equivalent for) on its parent
    link's collision geometry.

    PhysX combines two materials' friction by averaging by default, where
    MuJoCo takes the max; the combine mode is forced to "max" here (see
    `sim.isaac.objects.add_cube`'s matching fix on the object side) so a
    grasp/slip comparison isn't confounded by the two engines disagreeing
    on combine policy before either engine's grasp physics enters the
    picture.
    """
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    for pad in desc.contact_pads:
        parent_subpath = _geometry_subpath(stage, pad.parent_link)
        if parent_subpath is None:
            continue
        parent_path = f"/{stage.GetDefaultPrim().GetName()}/Geometry/{parent_subpath}"
        material_path = f"{parent_path}/{pad.name}_material"
        material = UsdShade.Material.Define(stage, material_path)
        physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics_material.CreateStaticFrictionAttr(pad.friction[0])
        physics_material.CreateDynamicFrictionAttr(pad.friction[0])
        physx_material = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
        physx_material.CreateFrictionCombineModeAttr("max")
        parent_prim = stage.GetPrimAtPath(parent_path)
        if parent_prim.IsValid():
            UsdShade.MaterialBindingAPI.Apply(parent_prim).Bind(
                material, materialPurpose="physics"
            )


def _geometry_subpath(stage: Any, mjcf_body_name: str) -> str | None:
    """Find `<link>_link`'s path under `.../Geometry/`, or `None`.

    The importer nests links under their parent (see `robots/so101
    /description.yaml`'s link-tree comment), so this is a search, not a
    fixed depth.
    """
    link_name = urdf_link_name(mjcf_body_name)
    root_name = stage.GetDefaultPrim().GetName()
    geometry_root = f"/{root_name}/Geometry"
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith(geometry_root) and prim.GetName() == link_name:
            return path[len(geometry_root) + 1 :]
    return None
