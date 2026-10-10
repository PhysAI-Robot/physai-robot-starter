import mujoco
import numpy as np
import pytest
from tests.conftest import requires_assets

pytestmark = [pytest.mark.acceptance, pytest.mark.assets]


@requires_assets
def test_a_plain_floor_has_one_colour_and_the_checker_has_a_texture():
    from physai.robots.registry import scene_defaults
    from physai.sim.mujoco import SingleCubePlaceSceneConfig

    def floor_material(style):
        model, _ = SingleCubePlaceSceneConfig(
            floor_style=style, **scene_defaults("so101")
        ).build_model()
        return model, model.mat("physai_grid")

    model, plain = floor_material("plain")
    assert plain.texid.max() < 0
    assert np.allclose(plain.rgba, [0.7569, 0.8020, 0.7765, 1.0], atol=1e-3)
    checker, material = floor_material("checker")
    assert material.texid.max() >= 0 and checker.ntex == model.ntex + 1


def test_an_unknown_floor_style_is_refused():
    from physai.robots.registry import scene_defaults
    from physai.sim.mujoco import SingleCubePlaceSceneConfig

    with pytest.raises(ValueError, match="floor_style"):
        SingleCubePlaceSceneConfig(
            floor_style="mud", **scene_defaults("so101")
        ).build_model()
