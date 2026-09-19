import threading

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


class ThreadRecordingRobot(FakeRobotPort):
    def __init__(self, spec: RobotSpec) -> None:
        super().__init__(spec)
        self.close_thread: int | None = None

    def close(self) -> None:
        self.close_thread = threading.get_ident()
        super().close()


def test_stop_closes_the_robot_on_the_physics_thread():
    """The renderer belongs to the thread that renders with it.

    Freeing it from the thread that called stop() crashed the process with an
    access violation on Windows, so the physics thread must close the robot.
    """
    spec = RobotSpec(
        name="test", kind="test", joint_names=("joint",), action_joint_names=("joint",)
    )
    robot = ThreadRecordingRobot(spec)
    host = NoPublishHost(robot, robot_name="test")

    host.start()
    physics_thread = host._thread
    host.stop()
    physics_thread.join(timeout=5.0)

    assert robot.closed
    assert robot.close_thread == physics_thread.ident
    assert robot.close_thread != threading.get_ident()


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_physics_thread_closes_the_robot_even_when_the_loop_raises():
    spec = RobotSpec(
        name="test", kind="test", joint_names=("joint",), action_joint_names=("joint",)
    )
    robot = ThreadRecordingRobot(spec)  # no .model, so the default publish raises
    host = SimulationHost(robot, robot_name="test")

    host.start()
    host._thread.join(timeout=5.0)

    assert robot.closed


def test_stop_closes_the_robot_directly_when_never_started():
    spec = RobotSpec(name="test", kind="test", joint_names=("joint",))
    robot = ThreadRecordingRobot(spec)
    host = SimulationHost(robot, robot_name="test")

    host.stop()

    assert robot.closed
    assert robot.close_thread == threading.get_ident()
