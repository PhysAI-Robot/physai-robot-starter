import numpy as np
import pytest

from conftest import requires_assets

pytestmark = requires_assets


@requires_assets
def test_scene_has_the_task_objects_and_cameras():
    import mujoco

    from physai.sim import build_model

    model, _ = build_model()
    names = lambda kind, n: {mujoco.mj_id2name(model, kind, i) for i in range(n)}

    assert {"front", "wrist"} <= names(mujoco.mjtObj.mjOBJ_CAMERA, model.ncam)
    assert {"cube", "table"} <= names(mujoco.mjtObj.mjOBJ_BODY, model.nbody)
    assert {"pad_static", "pad_moving"} <= names(mujoco.mjtObj.mjOBJ_GEOM, model.ngeom)
    assert {"gripperframe", "target_site"} <= names(mujoco.mjtObj.mjOBJ_SITE, model.nsite)


@requires_assets
def test_task_specific_scene_configs_have_separate_object_layouts():
    import mujoco

    from physai.sim import PickPlaceMinimalSceneConfig, SortingMinimalSceneConfig, build_model

    pick_model, _ = build_model(PickPlaceMinimalSceneConfig())
    sorting_model, _ = build_model(SortingMinimalSceneConfig())
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
    # An overlapping slab silently jams shoulder_pan against its force limit,
    # which looks like a broken IK solver rather than a broken scene.
    from physai.sim import SceneConfig

    cfg = SceneConfig()
    table_near_edge = cfg.table_pos[0] - cfg.table_size[0]
    assert table_near_edge > 0.06, "table must start clear of the base"


@requires_assets
def test_reset_is_deterministic_for_a_given_seed(env):
    a = env.reset(seed=7)
    cube_a = env.cube_pos.copy()
    b = env.reset(seed=7)
    np.testing.assert_allclose(a.joint_state.position, b.joint_state.position)
    np.testing.assert_allclose(cube_a, env.cube_pos)


@requires_assets
def test_step_tracks_commanded_joint_positions(env):
    from physai.contracts import Action, GripperCommand

    obs = env.reset(seed=0)
    target = obs.joint_state.position[:5] + np.array([0.2, 0.1, -0.1, 0.05, 0.0])
    for _ in range(120):
        obs, *_ = env.step(Action(joint_position=target,
                                  gripper=GripperCommand(position=0.5)))
    # If this fails, something in the scene is blocking the arm.
    np.testing.assert_allclose(obs.joint_state.position[:5], target, atol=0.02)


@requires_assets
def test_gripper_normalisation_round_trips(env):
    for n in (0.0, 0.25, 0.5, 1.0):
        assert env.joint_to_gripper(env.gripper_to_joint(n)) == pytest.approx(n)


@requires_assets
def test_ik_reaches_a_point_on_the_table(env):
    from physai.robots.so101.kinematics import TOP_DOWN

    obs = env.reset(seed=0)
    target = env.cube_pos + np.array([0.0, 0.0, 0.01])
    res = env.kin.ik(target, TOP_DOWN, q_init=obs.joint_state.position[:5])
    assert res.converged, f"pos_err={res.position_error} rot_err={res.orientation_error}"


@requires_assets
def test_ik_pinch_puts_the_object_between_the_jaws_not_on_the_site(env):
    from physai.contracts import Action, GripperCommand
    from physai.robots.so101.kinematics import TOP_DOWN

    obs = env.reset(seed=0)
    target = env.cube_pos.copy()
    res = env.kin.ik_pinch(target, TOP_DOWN, q_init=obs.joint_state.position[:5])
    for _ in range(150):
        obs, *_ = env.step(Action(joint_position=res.qpos,
                                  gripper=GripperCommand(position=0.55)))
    pinch = env.kin.pinch_center(env.data)
    site = obs.ee_pose.pose.position.as_array()
    assert np.linalg.norm(pinch - target) < 0.015
    # The site itself must NOT be on the object, or the fixed jaw is inside it.
    assert np.linalg.norm(site - target) > 0.008


@requires_assets
def test_sorting_scene_has_three_colored_cubes():
    import mujoco

    from physai.sim import SceneConfig, build_model

    model, _ = build_model(SceneConfig(num_cubes=3))
    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)}
    assert {"cube_red", "cube_blue", "cube_yellow"} <= names
    assert "cube" not in names


@requires_assets
def test_sorting_env_exposes_target_color_and_all_cube_positions():
    from physai.sim import EnvConfig, SceneConfig, SO101PickPlaceEnv

    e = SO101PickPlaceEnv(EnvConfig(
        scene=SceneConfig(num_cubes=3), task="sorting", render=False, max_steps=200,
    ))
    try:
        e.reset(seed=3)
        assert e.target_color in {"red", "blue", "yellow"}
        positions = e.cube_positions
        assert set(positions) == {"red", "blue", "yellow"}
        np.testing.assert_allclose(e.cube_pos, positions[e.target_color])
    finally:
        e.close()


@requires_assets
def test_top_down_approach_is_unreachable_high_above_the_table(env):
    """Documents a real limit of this 5-DoF arm rather than a solver bug."""
    from physai.robots.so101.kinematics import TOP_DOWN

    obs = env.reset(seed=0)
    high = env.cube_pos + np.array([0.0, 0.0, 0.12])
    strict = env.kin.ik(high, TOP_DOWN, q_init=obs.joint_state.position[:5])
    loose = env.kin.ik(high, None, q_init=obs.joint_state.position[:5])
    assert loose.position_error < 1e-2       # the point itself is reachable
    assert not strict.converged              # but not with the jaws pointing down
