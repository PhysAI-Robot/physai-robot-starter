import pytest

from physai.robots.so101.scene import scene_defaults
from physai.sim.mujoco import SingleCubePlaceSceneConfig, SortingMinimalSceneConfig
from physai.sim.workspace import (
    CUBE_FRICTION,
    FRONT_CAMERA_FOVY_DEG,
    TABLE_FRICTION,
    TARGET_RGBA,
    CubeSpec,
    WorkspaceConfig,
)


def test_a_bare_workspace_places_no_cubes():
    assert WorkspaceConfig().cubes() == ()


def test_the_single_cube_scene_places_one_cube_at_its_configured_pose():
    scene = SingleCubePlaceSceneConfig()
    assert scene.cubes() == (
        CubeSpec(
            "cube", scene.cube_pos, scene.cube_half, scene.cube_mass, scene.cube_rgba
        ),
    )


def test_the_sorting_scene_spaces_its_cubes_along_y():
    scene = SortingMinimalSceneConfig()
    cubes = scene.cubes()
    assert [cube.name for cube in cubes] == list(scene.cube_names)
    assert [cube.rgba for cube in cubes] == list(scene.cube_rgba)
    ys = [cube.position[1] for cube in cubes]
    assert ys == pytest.approx([scene.cube_pos[1] + 0.06 * i for i in range(3)])


def test_mismatched_sorting_names_and_colors_are_rejected():
    scene = SortingMinimalSceneConfig(cube_names=("a", "b"))
    with pytest.raises(ValueError, match="same length"):
        scene.cubes()


@pytest.mark.assets
def test_the_mujoco_scene_is_built_from_the_shared_constants():
    import mujoco

    model, _ = SingleCubePlaceSceneConfig(**scene_defaults()).build_model()

    def geom(name):
        return model.geom(name)

    assert tuple(geom("table_top").friction) == pytest.approx(TABLE_FRICTION)
    assert tuple(geom("cube_geom").friction) == pytest.approx(CUBE_FRICTION)
    assert tuple(geom("target_pad").rgba) == pytest.approx(TARGET_RGBA)
    assert model.camera("front").fovy[0] == pytest.approx(FRONT_CAMERA_FOVY_DEG)
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube") >= 0
