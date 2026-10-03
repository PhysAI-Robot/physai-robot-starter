"""Generic Isaac Sim world primitives: the Isaac analogue of
`sim.mujoco.scenes.common`'s studio lighting/floor and task-object geoms.

A bare URDF import carries no lights, floor, or workspace objects (URDF has
no such concept), unlike a MuJoCo scene MJCF, which is why this exists as
its own module rather than folding into `physai.sim.isaac.description` (robot
data) or `physai.sim.isaac.core` (stepping/rendering machinery).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..studio import (
    FLOOR_TILE_M,
    ISAAC_DOME_INTENSITY,
    ISAAC_KEY_INTENSITY,
    STUDIO_FLOOR_RGB1,
    STUDIO_FLOOR_RGB2,
)


def add_studio_lighting(
    stage: Any,
    *,
    dome_intensity: float = ISAAC_DOME_INTENSITY,
    key_intensity: float = ISAAC_KEY_INTENSITY,
    scale: float = 1.0,
) -> None:
    """Add a simple key/fill light pair so a camera render is not black.

    A bare URDF import has no lights at all (URDF has no lighting concept);
    without this, `IsaacSimulationCore.render_camera` returns all-zero
    frames. Not colour-matched to the MuJoCo web-viewer studio palette
    (`sim.mujoco.scenes.common.STUDIO_SKY_RGB`) yet — that match only matters once
    a dataset image comparison needs it. `scale` multiplies both intensities
    (a lighting-difficulty level).
    """
    from pxr import UsdLux

    if stage.GetPrimAtPath("/World_lights").IsValid():
        return
    dome = UsdLux.DomeLight.Define(stage, "/World_lights/dome")
    dome.CreateIntensityAttr(dome_intensity * scale)
    key = UsdLux.DistantLight.Define(stage, "/World_lights/key")
    key.CreateIntensityAttr(key_intensity * scale)
    key.CreateAngleAttr(1.0)


def checker_texture(path: Path) -> Path:
    """Write the 2 x 2 checker image `add_ground_plane` repeats (once per two
    tiles) and return its path. Row 0 is the top of the image, v = 1 in USD."""
    import imageio.v3 as iio
    import numpy as np

    light = np.round(np.asarray(STUDIO_FLOOR_RGB1) * 255).astype(np.uint8)
    dark = np.round(np.asarray(STUDIO_FLOOR_RGB2) * 255).astype(np.uint8)
    half = 64
    image = np.zeros((2 * half, 2 * half, 3), dtype=np.uint8)
    image[:half, :half] = light  # u < 1/2, v > 1/2: odd tile row, even column
    image[:half, half:] = dark
    image[half:, :half] = dark  # bottom-left: the square containing the origin
    image[half:, half:] = light
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, image)
    return path


def add_ground_plane(
    stage: Any, *, size: float = 5.0, texture_path: Path | None = None
) -> None:
    """Add the floor: an invisible collision slab with a checkered visual.

    The Isaac analogue of MuJoCo's `physai_floor` plane (checker squares of
    `FLOOR_TILE_M`, aligned to the same origin). The collision slab is a plain
    cube, as before; the visible floor is a flat mesh just above it, textured
    with `checker_texture(texture_path)`. Without a `texture_path` the floor
    stays the flat grey slab (used by tests that need no rendering).
    """
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

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
    if texture_path is None:
        return
    plane.CreateVisibilityAttr(UsdGeom.Tokens.invisible)

    extent = 4.0 * size
    mesh = UsdGeom.Mesh.Define(stage, f"{path}_visual")
    mesh.CreatePointsAttr(
        [Gf.Vec3f(-extent, -extent, 0.0), Gf.Vec3f(extent, -extent, 0.0)]
        + [Gf.Vec3f(extent, extent, 0.0), Gf.Vec3f(-extent, extent, 0.0)]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([Gf.Vec3f(0.0, 0.0, 1.0)] * 4)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    # One image repeat covers two tiles, so u = x / (2 * tile).
    scale = 1.0 / (2.0 * FLOOR_TILE_M)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    st.Set(
        Vt.Vec2fArray(
            [
                Gf.Vec2f(x * scale, y * scale)
                for x, y in (
                    (-extent, -extent),
                    (extent, -extent),
                    (extent, extent),
                    (-extent, extent),
                )
            ]
        )
    )

    material = UsdShade.Material.Define(stage, f"{path}_material")
    shader = UsdShade.Shader.Define(stage, f"{path}_material/surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    reader = UsdShade.Shader.Define(stage, f"{path}_material/st")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    texture = UsdShade.Shader.Define(stage, f"{path}_material/checker")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture_path))
    texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.ConnectableAPI(), "result"
    )
    texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.ConnectableAPI(), "rgb"
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)


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
