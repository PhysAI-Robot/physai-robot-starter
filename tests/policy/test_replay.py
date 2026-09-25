import numpy as np
import pytest
from conftest import requires_assets

pytestmark = pytest.mark.assets


@requires_assets
def test_replay_policy_reproduces_recorded_actions(env):
    from physai.policy import ReplayPolicy

    obs = env.reset(seed=2)
    recorded = np.tile(
        np.concatenate([obs.joint_state.position[:5], [env.gripper_to_joint(0.5)]]),
        (10, 1),
    )
    policy = ReplayPolicy(env, recorded)
    policy.reset(obs)
    action = policy.act(obs)
    np.testing.assert_allclose(action.joint_position, recorded[0][:5])
    assert action.gripper.clipped() == 0.5
