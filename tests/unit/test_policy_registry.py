def test_builtin_policies_are_discoverable():
    from physai.policy import available_policies

    assert {
        "constant",
        "constant_twist",
        "scripted",
        "visual_servo",
        "replay",
    } <= set(available_policies())


def test_lerobot_policy_registers_itself_when_its_research_module_is_imported():
    import research.imitation_learning.vla_adapter  # noqa: F401
    from physai.policy import available_policies

    assert "lerobot" in available_policies()


def test_constant_policy_is_created_through_registry():
    from physai.policy import ConstantPolicy, create_policy

    policy = create_policy("constant")

    assert isinstance(policy, ConstantPolicy)


def test_constant_twist_policy_emits_twist_action():
    import numpy as np

    from physai.contracts import JointState, Observation
    from physai.policy import create_policy

    observation = Observation(
        joint_state=JointState(
            name=("left_wheel", "right_wheel"),
            position=np.zeros(2),
            velocity=np.zeros(2),
            effort=np.zeros(2),
        )
    )

    action = create_policy("constant_twist").act(observation)

    assert action.mode == "twist"


def test_unknown_policy_lists_available_policies():
    import pytest

    from physai.policy import create_policy

    with pytest.raises(ValueError, match="constant"):
        create_policy("does-not-exist")


def test_constant_policy_holds_a_zero_twist_for_a_base_robot():
    from types import SimpleNamespace

    from physai.policy import ConstantTwistPolicy, create_policy
    from physai.robots import RobotSpec

    spec = RobotSpec(
        name="base",
        kind="mobile_base",
        joint_names=("wheel",),
        action_joint_names=("wheel",),
        capabilities=("base_velocity",),
    )

    policy = create_policy("constant", env=SimpleNamespace(robot_spec=spec))

    assert isinstance(policy, ConstantTwistPolicy)
