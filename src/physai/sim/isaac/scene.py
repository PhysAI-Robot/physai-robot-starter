"""Generic Isaac Sim world primitives: the Isaac analogue of
`sim.mujoco.scenes.common`'s studio lighting/floor and task-object geoms.

A bare URDF import carries no lights, floor, or workspace objects (URDF has
no such concept), unlike a MuJoCo scene MJCF, which is why this exists as
its own module rather than folding into `physai.sim.isaac.description` (robot
data) or `physai.sim.isaac.core` (stepping/rendering machinery).
"""

from __future__ import annotations

from typing import Any


def add_studio_lighting(stage: Any) -> None:
    """Add a simple key/fill light pair so a camera render is not black.

    A bare URDF import has no lights at all (URDF has no lighting concept);
    without this, `IsaacSimulationCore.render_camera` returns all-zero
    frames. Not colour-matched to the MuJoCo web-viewer studio palette
    (`sim.mujoco.scenes.common.STUDIO_SKY_RGB`) yet — that match only matters once
    a dataset image comparison needs it.
    """
    from pxr import UsdLux

    if stage.GetPrimAtPath("/World_lights").IsValid():
        return
    dome = UsdLux.DomeLight.Define(stage, "/World_lights/dome")
    dome.CreateIntensityAttr(300.0)
    key = UsdLux.DistantLight.Define(stage, "/World_lights/key")
    key.CreateIntensityAttr(500.0)
    key.CreateAngleAttr(1.0)


def add_ground_plane(stage: Any, *, size: float = 5.0) -> None:
    """Add a plain ground plane, the Isaac analogue of MuJoCo's
    `physai_floor` geom in `sim.mujoco.scenes.common.build_manipulation_spec`.
    """
    from pxr import Gf, UsdGeom, UsdPhysics

    path = "/World_ground"
    if stage.GetPrimAtPath(path).IsValid():
        return
    plane = UsdGeom.Cube.Define(stage, path)
    plane.CreateSizeAttr(1.0)
    # Translate before scale: USD applies the first-listed op last, so
    # listing scale first scaled the -0.005 z offset by 0.01 too, leaving the
    # slab's top face ~5 mm *above* z=0 -- objects spawned at z=half_size
    # started interpenetrating it and were pushed out on the first step, and
    # every resting height read 5 mm high.
    plane.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.005))
    plane.AddScaleOp().Set(Gf.Vec3f(size, size, 0.01))
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim())


def add_world_camera(
    stage: Any,
    path: str,
    *,
    position: tuple[float, float, float],
    quat_wxyz: tuple[float, float, float, float],
    fovy_deg: float,
    width: int,
    height: int,
) -> str:
    """A world-fixed camera, not attached to the robot's own articulation --
    unlike `sim.isaac.description.apply_cameras`'s link-mounted cameras
    (which move with the arm). The Isaac analogue of MuJoCo's own
    world-body camera (`sim.mujoco.scenes.common.build_manipulation_spec`'s
    "front" camera). `quat_wxyz` is already in USD's own convention
    (callers needing MuJoCo's `xyaxes` convention -- x/y axis vectors, not
    a quaternion -- convert with `mujoco.mju_mat2Quat` first, since
    `physai.sim.isaac` itself must not import `mujoco`).
    """
    from pxr import Gf, UsdGeom

    from .description import (
        DEFAULT_CLIPPING_RANGE_M,
        DEFAULT_VERTICAL_APERTURE_MM,
        fovy_to_focal_length,
    )

    cam = UsdGeom.Camera.Define(stage, path)
    cam.AddTranslateOp().Set(Gf.Vec3d(*position))
    w, x, y, z = quat_wxyz
    cam.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    vertical_aperture_mm = DEFAULT_VERTICAL_APERTURE_MM
    cam.CreateFocalLengthAttr(
        fovy_to_focal_length(fovy_deg, vertical_aperture_mm=vertical_aperture_mm)
    )
    cam.CreateVerticalApertureAttr(vertical_aperture_mm)
    cam.CreateHorizontalApertureAttr(vertical_aperture_mm * width / height)
    cam.CreateClippingRangeAttr(Gf.Vec2f(*DEFAULT_CLIPPING_RANGE_M))
    return path
