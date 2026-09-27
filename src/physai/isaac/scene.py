"""Generic Isaac Sim world primitives: the Isaac analogue of
`sim.scenes.common`'s studio lighting/floor and task-object geoms.

A bare URDF import carries no lights, floor, or workspace objects (URDF has
no such concept), unlike a MuJoCo scene MJCF, which is why this exists as
its own module rather than folding into `physai.isaac.description` (robot
data) or `physai.isaac.core` (stepping/rendering machinery).
"""

from __future__ import annotations

from typing import Any


def add_studio_lighting(stage: Any) -> None:
    """Add a simple key/fill light pair so a camera render is not black.

    A bare URDF import has no lights at all (URDF has no lighting concept);
    without this, `IsaacSimulationCore.render_camera` returns all-zero
    frames. Not colour-matched to the MuJoCo web-viewer studio palette
    (`sim.scenes.common.STUDIO_SKY_RGB`) yet — that match only matters once
    a dataset image comparison needs it.
    """
    from pxr import UsdLux

    if stage.GetPrimAtPath("/World_lights").IsValid():
        return
    dome = UsdLux.DomeLight.Define(stage, "/World_lights/dome")
    dome.CreateIntensityAttr(1000.0)
    key = UsdLux.DistantLight.Define(stage, "/World_lights/key")
    key.CreateIntensityAttr(3000.0)
    key.CreateAngleAttr(1.0)


def add_ground_plane(stage: Any, *, size: float = 5.0) -> None:
    """Add a plain ground plane, the Isaac analogue of MuJoCo's
    `physai_floor` geom in `sim.scenes.common.build_manipulation_spec`.
    """
    from pxr import Gf, UsdGeom, UsdPhysics

    path = "/World_ground"
    if stage.GetPrimAtPath(path).IsValid():
        return
    plane = UsdGeom.Cube.Define(stage, path)
    plane.CreateSizeAttr(1.0)
    plane.AddScaleOp().Set(Gf.Vec3f(size, size, 0.01))
    plane.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.005))
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
