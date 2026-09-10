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
def test_domain_randomization_is_seeded_bounded_and_restores_baseline():
    from physai.sim import (
        DomainRandomizationConfig,
        DomainRandomizationEngine,
        SceneConfig,
        build_model,
    )

    model, _ = build_model(SceneConfig(clutter_count=2))
    baseline_friction = model.geom_friction.copy()
    baseline_mass = model.body_mass.copy()
    baseline_lighting = model.light_diffuse.copy()
    disabled = DomainRandomizationEngine(
        model, DomainRandomizationConfig(enabled=False)
    )
    config = DomainRandomizationConfig(
        enabled=True,
        friction_scale=(0.8, 1.2),
        mass_scale=(0.9, 1.1),
        lighting_scale=(0.7, 1.3),
        camera_position_jitter=0.01,
    )
    engine = DomainRandomizationEngine(model, config)
    first = engine.apply(np.random.default_rng(12), seed=12)
    first_friction = model.geom_friction.copy()
    first_mass = model.body_mass.copy()
    first_lighting = model.light_diffuse.copy()

    second = engine.apply(np.random.default_rng(12), seed=12)
    assert first.as_dict() == second.as_dict()
    np.testing.assert_allclose(model.geom_friction, first_friction)
    np.testing.assert_allclose(model.body_mass, first_mass)
    np.testing.assert_allclose(model.light_diffuse, first_lighting)
    assert config.friction_scale[0] <= first.friction_scale <= config.friction_scale[1]
    assert config.mass_scale[0] <= first.mass_scale <= config.mass_scale[1]
    assert config.lighting_scale[0] <= first.lighting_scale <= config.lighting_scale[1]
    for offset in first.camera_position_offset.values():
        assert np.max(np.abs(offset)) <= config.camera_position_jitter
    assert set(first.clutter_position) == {"physai_clutter_0", "physai_clutter_1"}
    for x, y in first.clutter_position.values():
        assert config.clutter_x_range[0] <= x <= config.clutter_x_range[1]
        assert config.clutter_y_range[0] <= y <= config.clutter_y_range[1]

    metadata = disabled.apply(np.random.default_rng(99), seed=99)
    assert not metadata.enabled
    np.testing.assert_allclose(model.geom_friction, baseline_friction)
    np.testing.assert_allclose(model.body_mass, baseline_mass)
    np.testing.assert_allclose(model.light_diffuse, baseline_lighting)


@requires_assets
def test_domain_randomization_metadata_is_recorded_in_episode_info():
    from physai.contracts import Action
    from physai.robots.so101 import EnvConfig, SO101Env
    from physai.sim import DomainRandomizationConfig

    env = SO101Env(EnvConfig(
        render=False,
        domain_randomization=DomainRandomizationConfig(
            enabled=True,
            camera_position_jitter=0.005,
        ),
    ))
    try:
        observation = env.reset(seed=21)
        _, _, _, _, info = env.step(
            Action(joint_position=observation.joint_state.position[:5])
        )
        assert info["randomization"] == env.randomization_metadata.as_dict()
        assert info["randomization"]["enabled"] is True
        assert info["randomization"]["seed"] == 21
    finally:
        env.close()


@requires_assets
def test_calibrated_pads_replace_jaw_collision_meshes():
    import mujoco

    from physai.sim import build_model

    model, _ = build_model()
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
@pytest.mark.parametrize("offset", [
    (0.00, -0.03, 0.01),
    (0.01, 0.04, 0.01),
    (-0.02, 0.07, 0.01),
])
def test_ik_reaches_representative_targets_within_metrics(env, offset):
    from physai.robots.so101.kinematics import TOP_DOWN

    obs = env.reset(seed=0)
    target = env.cube_pos + np.asarray(offset)
    result = env.kin.ik(target, TOP_DOWN, q_init=obs.joint_state.position[:5])

    assert result.converged
    assert result.position_error <= 1e-3
    assert result.orientation_error <= 3e-2
    assert 0 < result.iterations <= 150
    assert np.isfinite(result.qpos).all()
    assert np.all(result.qpos >= env.kin.limits[:, 0])
    assert np.all(result.qpos <= env.kin.limits[:, 1])


@requires_assets
def test_ik_accepts_strict_orientation_target(env):
    from physai.robots.so101.kinematics import top_down_quat

    obs = env.reset(seed=0)
    result = env.kin.ik(
        env.cube_pos + np.array([0.0, 0.0, 0.01]),
        q_init=obs.joint_state.position[:5],
        target_quat_wxyz=top_down_quat(),
        pos_tol=1e-3,
        rot_tol=3e-2,
    )

    assert result.converged
    assert result.position_error <= 1e-3
    assert result.orientation_error <= 3e-2


@requires_assets
def test_ik_rejects_unreachable_target_without_unsafe_joint_command(env):
    from physai.robots.so101.kinematics import TOP_DOWN

    obs = env.reset(seed=0)
    result = env.kin.ik(
        env.cube_pos + np.array([0.0, 0.0, 0.12]),
        TOP_DOWN,
        q_init=obs.joint_state.position[:5],
    )

    assert not result.converged
    assert np.isfinite(result.qpos).all()
    assert np.all(result.qpos >= env.kin.limits[:, 0])
    assert np.all(result.qpos <= env.kin.limits[:, 1])


@requires_assets
def test_ik_collision_acceptance_allows_grasp_contact_but_rejects_table_contact(env):
    import mujoco
    from physai.robots.so101.kinematics import TOP_DOWN

    obs = env.reset(seed=0)
    safe_result = env.kin.ik(
        env.cube_pos + np.array([0.0, 0.0, 0.01]),
        TOP_DOWN,
        q_init=obs.joint_state.position[:5],
    )
    env.data.qpos[env.arm_qadr] = safe_result.qpos
    mujoco.mj_forward(env.model, env.data)
    assert env.kin.forbidden_contact_body_pairs(
        env.data,
        allowed_body_pairs=(("gripper", "cube"),),
    ) == ()

    low_result = env.kin.ik(
        [0.20, 0.02, env.table_top + 0.001],
        TOP_DOWN,
        q_init=obs.joint_state.position[:5],
    )
    assert low_result.converged
    env.data.qpos[env.arm_qadr] = low_result.qpos
    mujoco.mj_forward(env.model, env.data)
    assert ("gripper", "table") in env.kin.forbidden_contact_body_pairs(
        env.data,
        allowed_body_pairs=(("gripper", "cube"),),
    )


@requires_assets
@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_ik_rejects_non_finite_inputs(env, bad_value):
    obs = env.reset(seed=0)
    with pytest.raises(ValueError, match="finite"):
        env.kin.ik(
            [bad_value, 0.0, env.table_top],
            q_init=obs.joint_state.position[:5],
        )

    with pytest.raises(ValueError, match="finite"):
        env.kin.ik(
            env.cube_pos,
            [bad_value, 0.0, 0.0],
            q_init=obs.joint_state.position[:5],
        )


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
    from physai.robots.so101 import EnvConfig, SO101Env
    from physai.sim import SceneConfig
    from physai.tasks import TaskRuntime, create_task

    robot = SO101Env(EnvConfig(
        scene=SceneConfig(num_cubes=3), render=False, max_steps=200,
    ))
    e = TaskRuntime(robot, create_task("sorting"))
    try:
        e.reset(seed=3)
        assert e.target_color in {"red", "blue", "yellow"}
        positions = e.cube_positions
        assert set(positions) == {"red", "blue", "yellow"}
        np.testing.assert_allclose(e.cube_pos, positions[e.target_color])
    finally:
        e.close()


@requires_assets
def test_sorting_reset_is_deterministic_for_a_given_seed():
    from physai.robots.so101 import EnvConfig, SO101Env
    from physai.sim import SceneConfig

    robot = SO101Env(EnvConfig(
        scene=SceneConfig(num_cubes=3), render=False, max_steps=200,
    ))
    try:
        first = robot.reset(seed=17)
        first_positions = {
            color: position.copy()
            for color, position in robot.cube_positions.items()
        }
        first_target = robot.target_color

        second = robot.reset(seed=17)
        assert robot.target_color == first_target
        assert set(robot.cube_positions) == set(first_positions)
        for color, position in first_positions.items():
            np.testing.assert_allclose(robot.cube_positions[color], position)
        np.testing.assert_allclose(
            second.joint_state.position,
            first.joint_state.position,
        )
        assert second.sim_time == first.sim_time == 0.0
    finally:
        robot.close()


@requires_assets
def test_both_observation_cameras_carry_signal():
    """The wrist camera used to point away from the workspace.

    It rendered a pure black frame for the whole episode, so every recorded
    demonstration carried one informative view and one blank one while the
    dataset still advertised two cameras.
    """
    from physai.robots.so101 import EnvConfig, SO101Env
    from physai.sim import SceneConfig

    robot = SO101Env(EnvConfig(
        scene=SceneConfig(camera_width=128, camera_height=128),
        seed=0, render=True, max_steps=200,
    ))
    try:
        robot.reset(seed=0)
        for name in ("front", "wrist"):
            frame = robot.render_camera(name).astype(float)
            # At this pose the misaimed wrist camera measured std 6.4 against
            # 84.5 once aimed correctly, and the front camera sits at 74.8.
            assert frame.std() > 20.0, f"{name} camera is nearly uniform (std={frame.std()})"
    finally:
        robot.close()


@requires_assets
def test_wrist_camera_looks_toward_the_object_it_is_grasping():
    """Guards the sign error directly: the view axis pointed 180 degrees off."""
    import mujoco

    from physai.robots.so101 import EnvConfig, SO101Env
    from physai.tasks import TaskRuntime, create_task

    robot = SO101Env(EnvConfig(seed=0, render=False, max_steps=200))
    e = TaskRuntime(robot, create_task("pick_place"))
    try:
        e.reset(seed=0)
        cam = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
        # A MuJoCo camera looks along the negative z axis of its own frame.
        view = -robot.data.cam_xmat[cam].reshape(3, 3)[:, 2]
        to_cube = e.cube_pos - robot.data.cam_xpos[cam]
        to_cube /= np.linalg.norm(to_cube)
        assert float(view @ to_cube) > 0.5, "wrist camera faces away from the cube"
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
