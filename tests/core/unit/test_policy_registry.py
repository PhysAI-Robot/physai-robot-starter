from types import SimpleNamespace

import numpy as np
import pytest

from physai.contracts import JointState, Observation


def test_policies_are_discovered_and_research_ones_register_on_import():
    import research.imitation_learning.vla_adapter  # noqa: F401
    from physai.policy import available_policies, create_policy

    assert {
        "constant",
        "constant_twist",
        "scripted",
        "visual_servo",
        "replay",
    } <= set(available_policies())
    assert "lerobot" in available_policies()  # its research module registered it
    with pytest.raises(ValueError, match="constant"):
        create_policy("does-not-exist")  # the error lists what is available


def test_hold_policies_are_shaped_for_the_robots_action_space():
    from physai.policy import ConstantPolicy, ConstantTwistPolicy, create_policy
    from physai.robots import RobotSpec

    assert isinstance(create_policy("constant"), ConstantPolicy)

    observation = Observation(
        joint_state=JointState(
            name=("left_wheel", "right_wheel"),
            position=np.zeros(2),
            velocity=np.zeros(2),
            effort=np.zeros(2),
        )
    )
    assert create_policy("constant_twist").act(observation).mode == "twist"

    spec = RobotSpec(
        name="base",
        kind="mobile_base",
        joint_names=("wheel",),
        action_joint_names=("wheel",),
        capabilities=("base_velocity",),
    )
    policy = create_policy("constant", env=SimpleNamespace(robot_spec=spec))
    assert isinstance(policy, ConstantTwistPolicy)
