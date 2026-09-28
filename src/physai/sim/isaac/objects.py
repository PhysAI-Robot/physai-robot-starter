"""Dynamic (graspable) rigid-body task objects for an Isaac Sim stage.

The Isaac analogue of `sim.mujoco.scenes.common.add_cube` — but unlike this
project's other Isaac primitives (`scene.py`'s ground plane, the robot's
own already-imported URDF geometry), a graspable object needs a rigid body
and mass in addition to collision geometry. Nothing in this codebase
authored one before the grasp-hold parity tier (`ROADMAP.md`'s 2E, tier 3),
so this is unverified against any prior working example here; treat it the
way `sim.isaac.description`'s own docstring treats the rest of this
project's Isaac integration — checked against the real app, not derived
from a documented guarantee.
"""

from __future__ import annotations

from typing import Any


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
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(2.0 * half_size)
    cube.AddTranslateOp().Set(Gf.Vec3d(*position))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*rgba[:3])])
    prim = cube.GetPrim()

    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)

    material_path = f"{path}_material"
    material = UsdShade.Material.Define(stage, material_path)
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr(friction)
    physics_material.CreateDynamicFrictionAttr(friction)
    physx_material = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physx_material.CreateFrictionCombineModeAttr("max")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
    return path


def prim_world_position(stage: Any, path: str) -> Any:
    """A prim's world-space translation, as a plain 3-element array."""
    import numpy as np
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise KeyError(f"prim {path!r} not found on stage")
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    translation = matrix.ExtractTranslation()
    return np.array([translation[0], translation[1], translation[2]])


__all__ = ["add_cube", "prim_world_position"]
