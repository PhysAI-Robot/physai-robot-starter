from types import SimpleNamespace

import numpy as np
import pytest

from physai.contracts import JointState, Observation


def test_policies_are_discovered():
    from physai.policy import available_policies, create_policy

    assert {"constant", "constant_twist", "replay"} <= set(available_policies())
    with pytest.raises(ValueError, match="constant"):
        create_policy("does-not-exist")  # the error lists what is available


def test_a_robot_owned_policy_is_listed_for_its_robot_and_created_by_name():
    from physai.policy import available_policies

    assert "demo" in available_policies("so101")
    assert "demo" not in available_policies("turtlebot4")


def test_an_unknown_policy_error_points_at_the_manifests_plugins():
    from physai.policy import create_policy

    with pytest.raises(ValueError, match="plugins"):
        create_policy("scripted")


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
