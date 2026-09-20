import numpy as np
import pytest
from conftest import requires_assets

pytestmark = [pytest.mark.acceptance, pytest.mark.assets, pytest.mark.slow]


@requires_assets
def test_scene_has_the_task_objects_and_cameras():
    import mujoco

    from physai.robots.registry import scene_defaults
    from physai.sim import PickPlaceMinimalSceneConfig

    model, _ = PickPlaceMinimalSceneConfig(**scene_defaults("so101")).build_model()
    names = lambda kind, n: {mujoco.mj_id2name(model, kind, i) for i in range(n)}
    assert {"front", "wrist"} <= names(mujoco.mjtObj.mjOBJ_CAMERA, model.ncam)
    assert {"cube", "table"} <= names(mujoco.mjtObj.mjOBJ_BODY, model.nbody)
    assert {"pad_static", "pad_moving"} <= names(mujoco.mjtObj.mjOBJ_GEOM, model.ngeom)
    assert {"gripperframe", "target_site"} <= names(
        mujoco.mjtObj.mjOBJ_SITE, model.nsite
    )


@requires_assets
def test_calibrated_pads_replace_jaw_collision_meshes():
    import mujoco

    from physai.robots.registry import scene_defaults
    from physai.sim import PickPlaceMinimalSceneConfig

    model, _ = PickPlaceMinimalSceneConfig(**scene_defaults("so101")).build_model()
    jaw_bodies = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in ("gripper", "moving_jaw_so101_v1")
    }
    for geom_id, body_id in enumerate(model.geom_bodyid):
        if body_id not in jaw_bodies:
            continue
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
            assert model.geom_contype[geom_id] == 0
            assert model.geom_conaffinity[geom_id] == 0
    for name in ("pad_static", "pad_moving"):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert model.geom_contype[geom_id] != 0
        assert model.geom_conaffinity[geom_id] != 0


@requires_assets
def test_task_specific_scene_configs_have_separate_object_layouts():
    import mujoco

    from physai.robots.registry import scene_defaults
    from physai.sim import PickPlaceMinimalSceneConfig, SortingMinimalSceneConfig

    defaults = scene_defaults("so101")
    pick_model, _ = PickPlaceMinimalSceneConfig(**defaults).build_model()
    sorting_model, _ = SortingMinimalSceneConfig(**defaults).build_model()
    body_names = lambda model: {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index)
        for index in range(model.nbody)
    }
    assert "cube" in body_names(pick_model)
    assert not {"cube_red", "cube_blue", "cube_yellow"} & body_names(pick_model)
    assert {"cube_red", "cube_blue", "cube_yellow"} <= body_names(sorting_model)
    assert "cube" not in body_names(sorting_model)


@requires_assets
def test_table_does_not_intersect_the_robot_base():
    from physai.sim import PickPlaceMinimalSceneConfig

    cfg = PickPlaceMinimalSceneConfig()
    table_near_edge = cfg.table_pos[0] - cfg.table_size[0]
    assert table_near_edge > 0.06


@requires_assets
def test_sorting_scene_has_three_colored_cubes():
    import mujoco

    from physai.robots.registry import scene_defaults
    from physai.sim import SortingMinimalSceneConfig

    model, _ = SortingMinimalSceneConfig(**scene_defaults("so101")).build_model()
    names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(model.nbody)
    }
    assert {"cube_red", "cube_blue", "cube_yellow"} <= names
    assert "cube" not in names


@requires_assets
def test_sorting_env_exposes_target_color_and_all_cube_positions():
    from physai.robots.so101 import EnvConfig, SO101Env
    from physai.sim import SortingMinimalSceneConfig
    from physai.tasks import TaskRuntime, create_task

    robot = SO101Env(
        EnvConfig(scene=SortingMinimalSceneConfig(), render=False, max_steps=200)
    )
    env = TaskRuntime(robot, create_task("sorting"))
    try:
        env.reset(seed=3)
        assert env.target_color in {"red", "blue", "yellow"}
        positions = env.cube_positions
        assert set(positions) == {"red", "blue", "yellow"}
        np.testing.assert_allclose(env.cube_pos, positions[env.target_color])
    finally:
        env.close()


@requires_assets
def test_sorting_reset_is_deterministic_for_a_given_seed():
    from physai.robots.so101 import EnvConfig, SO101Env
    from physai.sim import SortingMinimalSceneConfig

    robot = SO101Env(
        EnvConfig(scene=SortingMinimalSceneConfig(), render=False, max_steps=200)
    )
    try:
        first = robot.reset(seed=17)
        first_positions = {
            color: position.copy() for color, position in robot.cube_positions.items()
        }
        first_target = robot.target_color
        second = robot.reset(seed=17)
        assert robot.target_color == first_target
        assert set(robot.cube_positions) == set(first_positions)
        for color, position in first_positions.items():
            np.testing.assert_allclose(robot.cube_positions[color], position)
        np.testing.assert_allclose(
            second.joint_state.position, first.joint_state.position
        )
        assert second.sim_time == first.sim_time == 0.0
    finally:
        robot.close()
