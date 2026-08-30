import numpy as np
import pytest

from physai.contracts import Action, JointState, Observation
from physai.robots import RobotSpec


class FakeRobot:
    def __init__(self, spec: RobotSpec) -> None:
        self.robot_spec = spec
        self.closed = False
        self.observation = Observation(
            joint_state=JointState(name=spec.joint_names, position=np.zeros(len(spec.joint_names)))
        )

    def reset(self, seed=None):
        del seed
        return self.observation

    def observe(self):
        return self.observation

    def send_action(self, action):
        self.robot_spec.validate_action(action)

    def step(self, action):
        self.send_action(action)
        return self.observation, 0.0, False, False, {}

    def close(self):
        self.closed = True


def test_robot_spec_rejects_wrong_joint_order():
    spec = RobotSpec(
        name="arm",
        kind="manipulator",
        joint_names=("a", "b"),
        action_joint_names=("a", "b"),
    )

    with pytest.raises(ValueError, match="expects joint order"):
        spec.validate_action(Action(joint_position=[0.0, 0.0], joint_names=("b", "a")))


def test_safety_rejects_stale_and_out_of_limit_actions():
    from physai.control import SafetyController

    spec = RobotSpec(
        name="arm",
        kind="manipulator",
        joint_names=("a",),
        action_joint_names=("a",),
        joint_limits={"a": (-1.0, 1.0)},
    )
    safety = SafetyController(spec, max_action_age=0.5)
    observation = Observation(joint_state=JointState(name=("a",), position=[0.0]))

    with pytest.raises(ValueError, match="stale"):
        safety.validate(observation, Action(joint_position=[0.0], stamp=9.0), now=10.0)
    with pytest.raises(ValueError, match="outside limits"):
        safety.validate(observation, Action(joint_position=[2.0]), now=10.0)


def test_runtime_rejects_incompatible_robot_task_before_episode(monkeypatch):
    from physai.runtime import composition

    spec = RobotSpec(
        name="mobile",
        kind="mobile_base",
        joint_names=("left_wheel", "right_wheel"),
        action_modes=("twist",),
        capabilities=("base_velocity",),
    )
    fake = FakeRobot(spec)
    monkeypatch.setattr(composition, "create_robot", lambda *args, **kwargs: fake)

    with pytest.raises(ValueError, match="does not support"):
        composition.create_runtime("mobile", task_name="pick_place")
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
    fake = FakeRobot(spec)
    monkeypatch.setattr(composition, "create_robot", lambda *args, **kwargs: fake)
    runtime = composition.create_runtime("arm", task_name="pick_place")
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
    fake = FakeRobot(spec)
    monkeypatch.setattr(composition, "create_robot", lambda *args, **kwargs: fake)

    runtime = composition.create_runtime("arm", task_name="pick_place")
    try:
        assert isinstance(runtime.robot, TaskRuntime)
        assert runtime.task.name == "pick_place"
    finally:
        runtime.close()


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
    fake = FakeRobot(spec)
    monkeypatch.setattr(composition, "create_robot", lambda *args, **kwargs: fake)

    runtime = composition.create_runtime("arm", task_name="pick_place")
    try:
        assert isinstance(runtime.robot, TaskRuntime)
        assert runtime.task.name == "pick_place"
    finally:
        runtime.close()
