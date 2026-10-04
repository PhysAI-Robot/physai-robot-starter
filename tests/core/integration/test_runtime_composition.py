import pytest

pytestmark = pytest.mark.integration

from physai.contracts import Action, JointState, Observation
from physai.robots import RobotSpec
from tests.core.support.fakes import FakeRobotPort


def test_actions_with_the_wrong_joint_order_or_stale_or_out_of_limit_are_rejected():
    from physai.control import SafetyController

    spec = RobotSpec(
        name="arm",
        kind="manipulator",
        joint_names=("a", "b"),
        action_joint_names=("a", "b"),
    )
    with pytest.raises(ValueError, match="expects joint order"):
        spec.validate_action(Action(joint_position=[0.0, 0.0], joint_names=("b", "a")))

    limited = RobotSpec(
        name="arm",
        kind="manipulator",
        joint_names=("a",),
        action_joint_names=("a",),
        joint_limits={"a": (-1.0, 1.0)},
    )
    safety = SafetyController(limited, max_action_age=0.5)
    observation = Observation(
        joint_state=JointState(
            name=("a",), position=[0.0], velocity=[0.0], effort=[0.0]
        )
    )
    with pytest.raises(ValueError, match="stale"):
        safety.validate(observation, Action(joint_position=[0.0], stamp=9.0), now=10.0)
    with pytest.raises(ValueError, match="outside limits"):
        safety.validate(observation, Action(joint_position=[2.0]), now=10.0)


def test_refused_actions_raise_the_safety_violation_an_evaluation_can_record():
    from physai.control import SafetyController, SafetyViolation

    spec = RobotSpec(
        name="arm",
        kind="manipulator",
        joint_names=("a",),
        action_joint_names=("a",),
        joint_limits={"a": (-1.0, 1.0)},
        max_joint_delta={"a": 0.1},
    )
    safety = SafetyController(spec, max_action_age=0.5)
    observation = Observation(
        joint_state=JointState(
            name=("a",), position=[0.0], velocity=[0.0], effort=[0.0]
        )
    )
    assert issubclass(SafetyViolation, ValueError)
    for action, now in (
        (Action(joint_position=[0.5]), 10.0),  # a step larger than allowed
        (Action(joint_position=[2.0]), 10.0),  # outside the joint limits
        (Action(joint_position=[0.0], stamp=9.0), 10.0),  # stale
    ):
        with pytest.raises(SafetyViolation):
            safety.validate(observation, action, now=now)
    # A malformed action is a programming error, not a safety event.
    with pytest.raises(ValueError) as error:
        safety.validate(observation, Action(joint_position=None), now=10.0)
    assert not isinstance(error.value, SafetyViolation)


def test_runtime_rejects_incompatible_robot_task_before_episode(monkeypatch):
    from physai.runtime import composition

    spec = RobotSpec(
        name="mobile",
        kind="mobile_base",
        joint_names=("left_wheel", "right_wheel"),
        action_modes=("twist",),
        capabilities=("base_velocity",),
    )
    fake = FakeRobotPort(spec)
    monkeypatch.setattr(composition, "create_robot", lambda *args, **kwargs: fake)

    with pytest.raises(ValueError, match="does not support"):
        composition.create_runtime("mobile", task_name="single_cube_fixed_place")
    assert fake.closed


def test_runtime_validates_action_before_forwarding(monkeypatch):
    from physai.runtime import composition

    spec = RobotSpec(
        name="arm",
        kind="manipulator",
        joint_names=("a",),
        action_joint_names=("a",),
        action_modes=("joint_position",),
        capabilities=("arm_kinematics", "gripper"),
        joint_limits={"a": (-1.0, 1.0)},
    )
    from physai.robots import DirectAdapter

    fake = FakeRobotPort(spec)
    adapter = DirectAdapter(fake)
    monkeypatch.setattr(composition, "create_robot", lambda *args, **kwargs: adapter)
    runtime = composition.create_runtime("arm", task_name="single_cube_fixed_place")
    try:
        runtime.reset()
        with pytest.raises(ValueError, match="outside limits"):
            runtime.step(Action(joint_position=[2.0]))
    finally:
        runtime.close()
    assert fake.closed


def test_runtime_composes_task_around_robot(monkeypatch):
    from physai.runtime import composition
    from physai.tasks import TaskRuntime

    spec = RobotSpec(
        name="arm",
        kind="manipulator",
        action_joint_names=("a",),
        action_modes=("joint_position",),
        capabilities=("arm_kinematics", "gripper"),
    )
    fake = FakeRobotPort(spec)
    monkeypatch.setattr(composition, "create_robot", lambda *args, **kwargs: fake)

    runtime = composition.create_runtime("arm", task_name="single_cube_fixed_place")
    try:
        assert isinstance(runtime.robot, TaskRuntime)
        assert runtime.task.name == "single_cube_fixed_place"
    finally:
        runtime.close()
