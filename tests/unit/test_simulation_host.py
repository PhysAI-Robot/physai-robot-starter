import numpy as np
import pytest

from physai.contracts import Action
from physai.robots import RobotSpec
from physai.web.runtime import SimulationHost
from tests.support.fakes import FakeRobotPort


def make_host() -> SimulationHost:
    spec = RobotSpec(
        name="test",
        kind="test",
        joint_names=("joint",),
        action_joint_names=("joint",),
    )
    return SimulationHost(FakeRobotPort(spec), robot_name="test")


class RecordingRobot(FakeRobotPort):
    def __init__(self, spec: RobotSpec) -> None:
        super().__init__(spec)
        self.seeds = []

    def reset(self, seed=None):
        self.seeds.append(seed)
        return super().reset(seed)


class NoPublishHost(SimulationHost):
    def _publish(self) -> None:
        pass


def test_control_lease_allows_one_client_and_releases_cleanly():
    host = make_host()
    action = Action(joint_position=np.array([0.2]))

    host.submit(action, source="browser")
    with pytest.raises(PermissionError, match="held by another client"):
        host.submit(action, source="desktop")

    host.release_control("browser")
    host.submit(action, source="desktop")


def test_reset_reuses_interactive_seed():
    spec = RobotSpec(name="test", kind="test", joint_names=("joint",))
    robot = RecordingRobot(spec)
    host = NoPublishHost(robot, robot_name="test", reset_seed=7)

    host.reset()
    host.reset()

    assert robot.seeds == [7, 7]


def test_auto_reset_reuses_interactive_seed():
    spec = RobotSpec(name="test", kind="test", joint_names=("joint",))
    robot = RecordingRobot(spec)
    host = NoPublishHost(robot, robot_name="test", reset_seed=7)

    host._reset_episode()

    assert robot.seeds == [7]


def test_control_lease_expiry_discards_queued_action():
    host = make_host()
    action = Action(joint_position=np.array([0.2]))

    host.submit(action, source="browser")
    host._control_deadline = 0.0

    assert host._latest_command() is None
