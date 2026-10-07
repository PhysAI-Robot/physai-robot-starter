"""Dynamic (graspable) rigid-body task objects for an Isaac Sim stage.

The Isaac analogue of `sim.mujoco.scenes.common.add_cube` — but unlike this
project's other Isaac primitives (`scene.py`'s ground plane, the robot's
own already-imported URDF geometry), a graspable object needs a rigid body
and mass in addition to collision geometry. Nothing in this codebase
authored one before the grasp-hold parity tier (sim-to-sim parity tier 3),
so this is unverified against any prior working example here; treat it the
way `sim.isaac.description`'s own docstring treats the rest of this
project's Isaac integration — checked against the real app, not derived
from a documented guarantee.
"""

from __future__ import annotations

from typing import Any


def _bind_matte_color(stage: Any, prim: Any, path: str, rgb: tuple[float, ...]) -> None:
    """Give `prim` a matte `UsdPreviewSurface` of this colour.

    A prim with only `displayColor` renders with the viewer's default,
    glossy-looking material, which washes a saturated red out to pink; MuJoCo's
    objects are matte, so a camera policy sees a different cube without this.
    """
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, f"{path}_look")
    shader = UsdShade.Shader.Define(stage, f"{path}_look/surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*rgb[:3])
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0, 0, 0)
    )
    # ior 1 removes the dielectric Fresnel reflection; with it, the bright sky
    # added a white sheen that washed the top of a red cube out to pale pink.
    shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _bind_friction_material(stage: Any, prim: Any, path: str, friction: float) -> None:
    """Bind a PhysX material with this friction (combine mode "max") to `prim`."""
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    material_path = f"{path}_material"
    material = UsdShade.Material.Define(stage, material_path)
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr(friction)
    physics_material.CreateDynamicFrictionAttr(friction)
    physx_material = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physx_material.CreateFrictionCombineModeAttr("max")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")


def add_static_box(
    stage: Any,
    path: str,
    *,
    position: tuple[float, float, float],
    half_extents: tuple[float, float, float],
    friction: float,
    rgba: tuple[float, float, float, float],
) -> str:
    """Add a static, collidable box centered at `position` (the Isaac analogue
    of a MuJoCo body-less box geom, e.g. the manipulation scene's table)."""
    from pxr import Gf, UsdGeom, UsdPhysics

    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    # Translate before scale; see `scene.add_ground_plane` for why the order matters.
    box.AddTranslateOp().Set(Gf.Vec3d(*position))
    box.AddScaleOp().Set(Gf.Vec3f(*(2.0 * h for h in half_extents)))
    box.CreateDisplayColorAttr([Gf.Vec3f(*rgba[:3])])
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    _bind_matte_color(stage, box.GetPrim(), path, rgba)
    _bind_friction_material(stage, box.GetPrim(), path, friction)
    return path


def add_target_pad(
    stage: Any,
    path: str,
    *,
    position: tuple[float, float, float],
    radius: float,
    rgba: tuple[float, float, float, float] = (0.2, 0.7, 0.35, 1.0),
) -> str:
    """Add a visual-only, non-colliding disc (MuJoCo's `target_pad`)."""
    from pxr import Gf, UsdGeom

    pad = UsdGeom.Cylinder.Define(stage, path)
    pad.CreateRadiusAttr(radius)
    pad.CreateHeightAttr(0.002)
    pad.AddTranslateOp().Set(Gf.Vec3d(*position))
    pad.CreateDisplayColorAttr([Gf.Vec3f(*rgba[:3])])
    _bind_matte_color(stage, pad.GetPrim(), path, rgba)
    return path


def add_cube(
    stage: Any,
    path: str,
    *,
    position: tuple[float, float, float],
    half_size: float,
    mass: float,
    friction: float,
    rgba: tuple[float, float, float, float] = (0.85, 0.25, 0.2, 1.0),
) -> str:
    """Add a dynamic, graspable cube centered at `position`. Returns `path`.

    `friction` is MuJoCo's sliding-friction component (`add_cube`'s own
    `friction[0]`, the only one a PhysX isotropic material has a direct
    equivalent for — mirrors `apply_contact_friction`'s reasoning). The
    friction-combine mode is forced to "max" for the same reason
    `apply_contact_friction` flags but leaves unset: PhysX's own default
    (average) would understate grip friction relative to MuJoCo's (max),
    confounding a slip comparison before either engine's grasp physics
    enters the picture.
    """
    from pxr import Gf, UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(2.0 * half_size)
    cube.AddTranslateOp().Set(Gf.Vec3d(*position))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*rgba[:3])])
    prim = cube.GetPrim()

    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
    _bind_matte_color(stage, prim, path, rgba)

    _bind_friction_material(stage, prim, path, friction)
    return path


__all__ = ["add_cube", "add_static_box", "add_target_pad"]
