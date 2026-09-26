"""Host is the one host class: single-robot (`for_robot`) and multi-robot
(`for_world`) sessions are both instances of it, exercised here side by side
since a single robot is just a session with one instance."""

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from physai.contracts import Action, Header, Pose, PoseStamped, Vector3
from physai.data import load_episode
from physai.robots import RobotSpec, shared_attach
from physai.sim import RobotInstanceConfig, SharedWorld
from physai.web.actions import action_from_payload
from physai.web.host import Host
from tests.core.support.fakes import FakeRobotPort

ROOT = Path(__file__).resolve().parents[3]
SO101_MODEL = ROOT / "assets" / "so101" / "so101_new_calib_camera.xml"
TURTLEBOT_MODEL = ROOT / "assets" / "turtlebot4" / "turtlebot4.xml"


def make_host() -> Host:
    spec = RobotSpec(
        name="test",
        kind="test",
        joint_names=("joint",),
        action_joint_names=("joint",),
    )
    return Host.for_robot(FakeRobotPort(spec), robot_name="test")


class RecordingRobot(FakeRobotPort):
    def __init__(self, spec: RobotSpec) -> None:
        super().__init__(spec)
        self.seeds = []

    def reset(self, seed=None):
        self.seeds.append(seed)
        return super().reset(seed)


class NoPublishHost(Host):
    def _publish(self) -> None:
        pass


def test_sync_observation_images_merges_the_async_camera_cache():
    """A vision-dependent policy must still see a camera frame in interactive
    mode even though the env itself never renders one there (camera_stride=0
    keeps physics stepping stutter-free; the async camera thread is the only
    renderer, and this is how its output reaches Observation.images)."""
    host = make_host()
    host._observation = host.robot.observe()
    assert host._observation.images == {}

    fake_frame = np.zeros((4, 4, 3), dtype=np.uint8)
    host._cameras.put(f"{host.robot_name}:front", fake_frame)

    host._sync_observation_images()

    assert "front" in host._observation.images
    frame = host._observation.images["front"]
    assert frame.camera_name == "front"
    np.testing.assert_array_equal(frame.data, fake_frame)


class FakeDebugPolicy:
    """A policy exposing the optional duck-typed debug-camera hook."""

    debug_camera_names = ("front:detections",)

    def __init__(self) -> None:
        self.acted = False

    def act(self, observation):
        self.acted = True
        return Action(joint_position=np.array([0.0]))

    def reset(self, observation, goal=None, instruction=None) -> None:
        pass

    @property
    def done(self) -> bool:
        return False

    def debug_frames(self):
        if not self.acted:
            return {}
        return {"front:detections": np.zeros((2, 2, 3), dtype=np.uint8)}


def test_a_policys_debug_cameras_are_listed_and_published_only_with_a_policy():
    spec = RobotSpec(
        name="test", kind="test", joint_names=("joint",), action_joint_names=("joint",)
    )
    policy = FakeDebugPolicy()
    policy.acted = True
    with_policy = Host.for_robot(FakeRobotPort(spec), robot_name="test", policy=policy)
    without_policy = make_host()

    assert with_policy.list_robots()[0]["cameras"] == ["front:detections"]
    assert without_policy.list_robots()[0]["cameras"] == []

    with_policy._publish_debug_frames()
    without_policy._publish_debug_frames()

    np.testing.assert_array_equal(
        with_policy._cameras.get("test:front:detections"),
        np.zeros((2, 2, 3), dtype=np.uint8),
    )
    assert without_policy._cameras.with_prefix("") == {}


def test_control_lease_allows_one_client_and_releases_cleanly():
    host = make_host()
    action = Action(joint_position=np.array([0.2]))

    host.submit(action, source="browser")
    with pytest.raises(PermissionError, match="controlled by another client"):
        host.submit(action, source="desktop")

    host.release_control("browser")
    host.submit(action, source="desktop")


def test_resets_reuse_the_interactive_seed():
    spec = RobotSpec(name="test", kind="test", joint_names=("joint",))
    robot = RecordingRobot(spec)
    host = NoPublishHost.for_robot(robot, robot_name="test", reset_seed=7)

    host.reset()
    host.reset()
    host._reset_episode()  # the automatic reset at the end of a policy episode

    assert robot.seeds == [7, 7, 7]


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
    host = NoPublishHost.for_robot(robot, robot_name="test")

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
    host = Host.for_robot(robot, robot_name="test")

    host.start()
    host._thread.join(timeout=5.0)

    assert robot.closed


def test_stop_closes_the_robot_directly_when_never_started():
    spec = RobotSpec(name="test", kind="test", joint_names=("joint",))
    robot = ThreadRecordingRobot(spec)
    host = Host.for_robot(robot, robot_name="test")

    host.stop()

    assert robot.closed
    assert robot.close_thread == threading.get_ident()


# -- multi-robot (Host.for_world): a session is just a world with N instances --

pytestmark_shared = pytest.mark.skipif(
    not SO101_MODEL.exists() or not TURTLEBOT_MODEL.exists(),
    reason="robot assets are not available",
)


def _heterogeneous_configs() -> tuple[RobotInstanceConfig, ...]:
    return (
        RobotInstanceConfig("arm_1", SO101_MODEL, "so101", position=(0.3, 0.0, 0.0)),
        RobotInstanceConfig(
            "base_1", TURTLEBOT_MODEL, "turtlebot4", position=(-0.3, 0.0, 0.1)
        ),
    )


@pytestmark_shared
def test_a_shared_host_lists_instances_routes_actions_and_stops_safely():
    configs = _heterogeneous_configs()
    world = SharedWorld(configs, control_hz=10.0, shared_attach=shared_attach)
    host = Host.for_world(world, configs)

    robots = {entry["name"]: entry for entry in host.list_robots()}
    assert robots["arm_1"]["kind"] == "fixed_base_manipulator"
    assert robots["arm_1"]["cameras"] == ["front", "wrist"]
    assert robots["base_1"]["kind"] == "mobile_base"
    assert robots["base_1"]["cameras"] == []

    host.reset()
    host.submit(
        action_from_payload({"mode": "twist", "linear": {"x": 0.01}, "angular": {}}),
        instance_id="arm_1",
    )
    host.submit(
        action_from_payload(
            {"mode": "twist", "linear": {"x": 0.1}, "angular": {"z": 0.2}}
        ),
        instance_id="base_1",
    )
    with host.physics_lock:
        host._tick_shared()
        host._publish()

    state = host.latest_state()
    assert state is not None
    assert state["step"] == 1
    owners = {item["instance_id"] for item in state["geometries"]}
    assert {"arm_1", "base_1"} <= owners

    host.stop()  # never started: must be safe
    assert host._thread is None


class QposRobot(FakeRobotPort):
    """A fake robot exposing `data.qpos`, as MuJoCo-backed robots do."""

    def __init__(self, spec: RobotSpec) -> None:
        super().__init__(spec)
        self.data = SimpleNamespace(qpos=np.array([0.1, 0.2, 0.3]), time=0.0)


def make_recording_host(record_dir, host_cls=Host) -> Host:
    spec = RobotSpec(
        name="test",
        kind="test",
        joint_names=("joint",),
        action_joint_names=("joint",),
    )
    host = host_cls.for_robot(QposRobot(spec), robot_name="test", record_dir=record_dir)
    host._observation = host.robot.observe()
    host._hold_action_value = host._hold_action()
    return host


def test_recording_saves_tagged_episodes_and_discards_on_request(tmp_path):
    host = make_recording_host(tmp_path)

    host.start_recording()
    for _ in range(3):
        host._tick_single()
    assert host.recording_status()["frames"] == 3
    host.stop_recording(True)
    host.start_recording()
    host._tick_single()
    host.stop_recording(False)
    host.start_recording()
    host._tick_single()
    host.stop_recording(None)  # discard the take

    status = host.recording_status()
    assert (status["episodes_saved"], status["successes"], status["active"]) == (
        2,
        1,
        False,
    )
    assert status["discarded"] == 1
    assert sorted(path.name for path in tmp_path.glob("*.npz")) == [
        "episode_00000.npz",
        "episode_00001.npz",
    ]
    episode = load_episode(tmp_path / "episode_00000.npz")
    assert episode["observation.state"].shape == (3, 1)
    np.testing.assert_allclose(
        episode["observation.environment_state"], [[0.1, 0.2, 0.3]] * 3
    )
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert [e["success"] for e in meta["episodes"]] == [True, False]
    assert meta["episodes"][0]["length"] == 3


def test_reset_discards_an_in_progress_recording(tmp_path):
    host = make_recording_host(tmp_path, NoPublishHost)
    host.start_recording()
    host._tick_single()

    host.reset()

    status = host.recording_status()
    assert (status["active"], status["discarded"]) == (False, 1)
    assert "reset" in status["error"]


def test_new_session_continues_numbering_an_existing_dataset(tmp_path):
    first = make_recording_host(tmp_path)
    first.start_recording()
    first._tick_single()
    first.stop_recording(True)

    second = make_recording_host(tmp_path)
    assert second.recording_status()["episodes_saved"] == 1
    second.start_recording()
    second._tick_single()
    second.stop_recording(True)

    assert (tmp_path / "episode_00001.npz").exists()
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["num_episodes"] == 2


def test_recording_state_is_merged_into_the_live_snapshot(tmp_path):
    host = make_recording_host(tmp_path)
    host._state = {"type": "state"}

    host.set_paused(True)
    host.start_recording()
    state = host.latest_state()

    assert state["paused"] is True
    assert state["recording"]["active"] is True


def test_recording_misuse_is_reported(tmp_path):
    plain = make_host()
    assert plain.recording_status() == {"enabled": False}
    with pytest.raises(ValueError, match="--record-dir"):
        plain.start_recording()

    host = make_recording_host(tmp_path)
    with pytest.raises(RuntimeError, match="not recording"):
        host.stop_recording(True)

    host.start_recording()
    with pytest.raises(RuntimeError, match="no frames"):
        host.stop_recording(True)
    assert host.recording_status()["episodes_saved"] == 0
    assert host.recording_status()["active"] is False


def test_recording_waits_for_every_camera_before_recording_frames(tmp_path):
    spec = RobotSpec(
        name="test",
        kind="test",
        joint_names=("joint",),
        action_joint_names=("joint",),
        camera_frames={"front": "camera_front"},
    )
    host = Host.for_robot(QposRobot(spec), robot_name="test", record_dir=tmp_path)
    host._observation = host.robot.observe()
    host._hold_action_value = host._hold_action()

    host.start_recording()
    host._tick_single()
    status = host.recording_status()
    assert (status["frames"], status["waiting_for"]) == (0, ["front"])

    host._cameras.put("test:front", np.full((4, 4, 3), 7, dtype=np.uint8))
    host._tick_single()
    host._tick_single()
    assert host.recording_status()["frames"] == 2
    host.stop_recording(True)

    images = load_episode(tmp_path / "episode_00000.npz")["observation.images.front"]
    assert images.shape == (2, 4, 4, 3)


class ToolPoseKinematics:
    """Kinematics offering the optional `tool_pose` extension."""

    def tool_pose(self, data):
        return PoseStamped(
            pose=Pose(position=Vector3(1.0, 2.0, 3.5)),
            header=Header(frame_id="base"),
        )


def test_ee_pose_payload_reports_the_observation_pose_or_the_tool_pose():
    host = make_host()
    assert host._ee_pose_payload() is None

    host._observation = host.robot.observe()
    assert host._observation.ee_pose is None
    assert host._ee_pose_payload() is None

    host._observation.ee_pose = PoseStamped(
        pose=Pose(position=Vector3(0.1, 0.2, 0.3)),
        header=Header(frame_id="base"),
    )
    assert host._ee_pose_payload() == {
        "frame_id": "base",
        "position": [0.1, 0.2, 0.3],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "reference": "ee_pose",
    }

    # a robot whose kinematics offers tool_pose wins over the observation's
    host.robot.kin = ToolPoseKinematics()
    host.robot.data = SimpleNamespace()
    assert host._ee_pose_payload() == {
        "frame_id": "base",
        "position": [1.0, 2.0, 3.5],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "reference": "tool",
    }


def test_camera_frames_are_served_from_the_feed_as_copies():
    host = make_host()
    frame = np.full((4, 4, 3), 9, dtype=np.uint8)
    host._cameras.put("test:front", frame)

    served = host.camera_image("front")
    served[:] = 0

    np.testing.assert_array_equal(host.camera_image("front"), frame)
    assert host.camera_jpeg("front").startswith(b"\xff\xd8")  # a JPEG
    with pytest.raises(ValueError, match="unknown camera 'side'"):
        host.camera_image("side")


def test_a_twist_jog_needs_a_resolver_the_robot_registered():
    from physai.robots.registry import RobotDescriptor, register_embodiment

    plain = make_host()
    plain._observation = plain.robot.observe()
    with pytest.raises(ValueError, match="twist jog is not available"):
        plain.submit(
            action_from_payload({"mode": "twist", "linear": {"x": 0.1}, "angular": {}})
        )

    calls = []

    def resolver(twist, joint_state, gripper):
        calls.append(twist.linear.x)
        return Action(joint_position=np.array([0.5]), gripper=gripper)

    register_embodiment(
        "_fake_jog_robot",
        RobotDescriptor(
            factory=lambda **_: None, kind="fake", jog=lambda robot: resolver
        ),
    )
    spec = RobotSpec(
        name="_fake_jog_robot",
        kind="fake",
        joint_names=("joint",),
        action_joint_names=("joint",),
    )
    host = Host.for_robot(FakeRobotPort(spec), robot_name="_fake_jog_robot")
    host._observation = host.robot.observe()

    host.submit(
        action_from_payload({"mode": "twist", "linear": {"x": 0.25}, "angular": {}})
    )

    assert calls == [0.25]
    assert host._latest_command("_fake_jog_robot").joint_position[0] == 0.5
