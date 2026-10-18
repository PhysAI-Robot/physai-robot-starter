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


def test_traits_come_from_the_registration_generic_or_robot_owned():
    from physai.contracts import PolicyTraits
    from physai.policy import policy_traits

    assert policy_traits("constant").simulators == ("mujoco", "isaac", "newton")
    assert policy_traits("demo", "so101") == PolicyTraits()
    with pytest.raises(ValueError, match="unknown policy"):
        policy_traits("demo")  # robot-owned: needs the robot to resolve


def test_a_plugin_policy_declares_its_traits_and_a_policy_has_no_extra_metrics():
    from physai.contracts import PolicyTraits
    from physai.policy import ConstantPolicy, policy_traits
    from physai.robots.registry import register_robot_policy

    traits = PolicyTraits(needs_images=True, simulators=("mujoco", "isaac"))
    register_robot_policy(
        "so101", "physai_test_traits", lambda **_: ConstantPolicy(), traits=traits
    )
    assert policy_traits("physai_test_traits", "so101") == traits
    assert ConstantPolicy().episode_metrics() == {}
    assert ConstantPolicy().training_seeds is None
