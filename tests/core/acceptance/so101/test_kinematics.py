import numpy as np
import pytest
from conftest import requires_assets

pytestmark = [pytest.mark.acceptance, pytest.mark.assets, pytest.mark.slow]


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
        obs, *_ = env.step(
            Action(joint_position=target, gripper=GripperCommand(position=0.5))
        )
    np.testing.assert_allclose(obs.joint_state.position[:5], target, atol=0.02)


@requires_assets
def test_gripper_normalisation_round_trips(env):
    for normalized in (0.0, 0.25, 0.5, 1.0):
        assert env.joint_to_gripper(env.gripper_to_joint(normalized)) == pytest.approx(
            normalized
        )


@requires_assets
def test_ik_reaches_a_point_on_the_table(env):
    from physai.robots.so101.kinematics import TOP_DOWN

    obs = env.reset(seed=0)
    target = env.cube_pos + np.array([0.0, 0.0, 0.01])
    result = env.kin.ik(target, TOP_DOWN, q_init=obs.joint_state.position[:5])
    assert result.converged, (
        f"pos_err={result.position_error} rot_err={result.orientation_error}"
    )


@requires_assets
def test_ik_reaches_representative_targets_within_metrics(env):
    from physai.robots.so101.kinematics import TOP_DOWN

    obs = env.reset(seed=0)
    for offset in [(0.00, -0.03, 0.01), (0.01, 0.04, 0.01), (-0.02, 0.07, 0.01)]:
        target = env.cube_pos + np.asarray(offset)
        result = env.kin.ik(target, TOP_DOWN, q_init=obs.joint_state.position[:5])
        assert result.converged, offset
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
    assert (
        env.kin.forbidden_contact_body_pairs(
            env.data,
            allowed_body_pairs=(("gripper", "cube"),),
        )
        == ()
    )
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
def test_ik_rejects_non_finite_inputs(env):
    obs = env.reset(seed=0)
    for bad_value in (np.nan, np.inf):
        with pytest.raises(ValueError, match="finite"):
            env.kin.ik(
                [bad_value, 0.0, env.table_top], q_init=obs.joint_state.position[:5]
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
    result = env.kin.ik_pinch(target, TOP_DOWN, q_init=obs.joint_state.position[:5])
    for _ in range(150):
        obs, *_ = env.step(
            Action(joint_position=result.qpos, gripper=GripperCommand(position=0.55))
        )
    pinch = env.kin.pinch_center(env.data)
    site = obs.ee_pose.pose.position.as_array()
    assert np.linalg.norm(pinch - target) < 0.015
    assert np.linalg.norm(site - target) > 0.008


@requires_assets
def test_top_down_approach_is_unreachable_high_above_the_table(env):
    from physai.robots.so101.kinematics import TOP_DOWN

    obs = env.reset(seed=0)
    high = env.cube_pos + np.array([0.0, 0.0, 0.12])
    strict = env.kin.ik(high, TOP_DOWN, q_init=obs.joint_state.position[:5])
    loose = env.kin.ik(high, None, q_init=obs.joint_state.position[:5])
    assert loose.position_error < 1e-2
    assert not strict.converged
