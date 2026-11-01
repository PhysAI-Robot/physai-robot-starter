import pytest
from tests.conftest import requires_assets, requires_viperx300s_assets

from physai.robots.reach import GraspReach


def _so101():
    from physai.robots.so101 import EnvConfig, SO101Env

    return SO101Env(EnvConfig(render=False))


def _viperx300s():
    from physai.robots.viperx300s import EnvConfig, ViperX300SEnv

    return ViperX300SEnv(EnvConfig(render=False))


@pytest.mark.parametrize(
    ("make_env", "near"),
    [
        pytest.param(_so101, (0.20, 0.0), marks=requires_assets),
        pytest.param(_viperx300s, (0.30, 0.0), marks=requires_viperx300s_assets),
    ],
)
def test_grasp_reach_separates_a_reachable_point_from_one_far_outside(make_env, near):
    env = make_env()
    try:
        q0 = env.reset().joint_state.position
        assert env.grasp_reach(*near, q_init=q0).grasp
        assert env.grasp_reach(*near, q_init=q0, hover=0.0, carry=0.0).ok
        assert env.grasp_reach(2.0, 0.0, q_init=q0) == GraspReach(False, False, False)
    finally:
        env.close()


@requires_viperx300s_assets
def test_the_viperx300s_grasps_top_down_only():
    env = _viperx300s()
    try:
        with pytest.raises(ValueError, match="top-down only"):
            env.grasp_reach(0.30, 0.0, tilt=60.0)
    finally:
        env.close()
