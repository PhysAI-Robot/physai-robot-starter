"""The scene Newton draws matches what MuJoCo draws: no collision-only geoms, textures as colours.

No GPU and no Newton import: `drawable_scene_mjcf` only rewrites the MJCF.
"""

import mujoco
import numpy as np


def _spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    texture = spec.add_texture(
        name="flat",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
        rgb1=[0.2, 0.4, 0.6],
        width=4,
        height=4,
    )
    assert texture is not None
    material = spec.add_material(name="painted")
    textures = list(material.textures)
    textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "flat"
    material.textures = textures
    material.rgba = [1.0, 1.0, 1.0, 1.0]
    body = spec.worldbody
    body.add_geom(
        name="visible",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.1] * 3,
        material="painted",
        group=2,
    )
    body.add_geom(
        name="collision_copy", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1] * 3, group=3
    )
    body.add_geom(name="plain", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1] * 3, group=0)
    return spec


def test_drawable_scene_drops_collision_geoms_and_bakes_texture_colour(tmp_path):
    from physai.sim.newton.render import drawable_scene_mjcf

    spec = _spec()
    model = spec.compile()
    with drawable_scene_mjcf(spec, model, tmp_path) as path:
        drawn = mujoco.MjSpec.from_file(path).compile()
        names = {
            mujoco.mj_id2name(drawn, mujoco.mjtObj.mjOBJ_GEOM, g)
            for g in range(drawn.ngeom)
        }
    assert names == {"visible", "plain"}
    assert drawn.mat_texid[0][mujoco.mjtTextureRole.mjTEXROLE_RGB] == -1
    np.testing.assert_allclose(drawn.mat_rgba[0], [0.2, 0.4, 0.6, 1.0], atol=0.01)
    assert list(tmp_path.iterdir()) == []  # the temporary files are gone
    assert model.ngeom == 3  # the env's own model is untouched
