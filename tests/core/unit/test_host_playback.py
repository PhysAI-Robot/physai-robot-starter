"""Episode playback on the paused Host (docs/adr/0009): restore recorded qpos,
step/scrub/play, and hand the live world back on exit."""

import mujoco
import numpy as np
import pytest

from physai.contracts import Action
from physai.data import load_episode
from physai.robots import RobotSpec
from physai.web.host import Host
from tests.core.support.fakes import FakeRobotPort

XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="arm" pos="0 0 0.5">
      <joint name="hinge" type="hinge" axis="0 0 1"/>
      <geom type="capsule" size="0.01 0.05"/>
    </body>
    <body name="cube" pos="0.3 0 0.05">
      <freejoint/>
      <geom name="pad_static" type="box" size="0.05 0.05 0.05" mass="0.5"/>
    </body>
  </worldbody>
</mujoco>
"""
FRAMES = 5


class MujocoFakeRobot(FakeRobotPort):
    """A fake robot over a real model whose joint drifts, so frames differ."""

    def __init__(self, spec: RobotSpec) -> None:
        super().__init__(spec)
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        self.step_count = 0

    def step(self, action):
        self.data.qpos[0] += 0.1
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        return super().step(action)


def make_host(record_dir) -> Host:
    spec = RobotSpec(
        name="test", kind="test", joint_names=("joint",), action_joint_names=("joint",)
    )
    host = Host.for_robot(
        MujocoFakeRobot(spec), robot_name="test", record_dir=record_dir
    )
    host._observation = host.robot.observe()
    host._hold_action_value = host._hold_action()
    return host


def record_episode(host: Host, success: bool = True) -> None:
    host.start_recording()
    for _ in range(FRAMES):
        host._tick_single()
    host.stop_recording(success)


def recorded_states(record_dir) -> np.ndarray:
    return load_episode(record_dir / "episode_00000.npz")[
        "observation.environment_state"
    ]


def test_playback_shows_recorded_frames_and_exit_restores_the_live_world(tmp_path):
    host = make_host(tmp_path)
    record_episode(host, success=False)
    for _ in range(3):
        host._tick_single()  # the live world moves on after the recording
    live_qpos = host.data.qpos.copy()
    states = recorded_states(tmp_path)

    (episode,) = host.list_episodes()
    assert (episode["file"], episode["success"], episode["playable"]) == (
        "episode_00000.npz",
        False,
        True,
    )
    assert episode["length"] == FRAMES

    host.load_episode("episode_00000.npz")

    assert host.paused
    assert host.latest_state()["playback"]["active"] is True
    np.testing.assert_allclose(host.data.qpos, states[0])
    host.seek(3)
    np.testing.assert_allclose(host.data.qpos, states[3])
    host.seek(1, relative=True)
    np.testing.assert_allclose(host.data.qpos, states[4])
    host.seek(99)  # out-of-range scrubs clamp to the last frame
    assert host.latest_state()["playback"]["frame"] == FRAMES - 1

    # scrubbing keeps playback running; stepping pauses it
    host.set_playback(True)
    host.seek(2)
    assert host.latest_state()["playback"]["playing"] is True
    host.seek(1, relative=True)
    assert host.latest_state()["playback"]["playing"] is False

    host.exit_playback()

    np.testing.assert_allclose(host.data.qpos, live_qpos)
    assert host.latest_state()["playback"] == {"active": False}
    assert host.paused  # exiting leaves the world paused


def test_playback_follows_wall_time_at_the_selected_speed(tmp_path):
    host = make_host(tmp_path)
    record_episode(host)
    host.load_episode("episode_00000.npz")
    tick = 1.0 / host.control_hz  # the recording's fps is the control rate

    host.set_playback(True, speed=2.0)
    host._advance_playback(100.0)  # the first tick only starts the clock
    host._advance_playback(100.0 + tick)
    assert host.latest_state()["playback"]["frame"] == 2
    host._advance_playback(100.0 + 2 * tick)
    assert host.latest_state()["playback"]["frame"] == 4
    host._advance_playback(100.0 + 3 * tick)
    status = host.latest_state()["playback"]
    assert (status["frame"], status["playing"]) == (FRAMES - 1, False)

    host.set_playback(True)  # replays from the start once it has finished
    assert host.latest_state()["playback"]["frame"] == 0
    host.set_playback(True, speed=0.5)
    host._advance_playback(200.0)
    host._advance_playback(200.0 + tick)
    host._advance_playback(200.0 + 2 * tick)
    assert host.latest_state()["playback"]["frame"] == 1

    # a slow loop still plays 1x in real time, and a stall is capped
    long = make_host(tmp_path / "long")
    long.start_recording()
    for _ in range(60):
        long._tick_single()
    long.stop_recording(True)
    long.load_episode("episode_00000.npz")
    long.set_playback(True)
    long._advance_playback(0.0)
    long._advance_playback(0.1)  # 0.1 s between ticks: 3 frames
    assert long.latest_state()["playback"]["frame"] == 3
    long._advance_playback(5.1)  # a 5 s stall advances by the 0.25 s cap: 7 frames
    assert long.latest_state()["playback"]["frame"] == 3 + 7


def test_playback_rejects_live_control_until_exited(tmp_path):
    host = make_host(tmp_path)
    record_episode(host)
    host.load_episode("episode_00000.npz")

    with pytest.raises(ValueError, match="exit playback"):
        host.submit(Action(joint_position=np.array([0.0])))
    with pytest.raises(ValueError, match="exit playback"):
        host.reset()
    with pytest.raises(ValueError, match="exit playback"):
        host.set_paused(False)
    with pytest.raises(ValueError, match="exit playback"):
        host.start_recording()

    host.exit_playback()
    host.set_paused(False)
    host.start_recording()


def test_loading_and_controlling_playback_reject_misuse(tmp_path):
    host = make_host(tmp_path)

    for call in (
        lambda: host.seek(0),
        lambda: host.set_playback(True),
        host.exit_playback,
    ):
        with pytest.raises(ValueError, match="no episode"):
            call()

    record_episode(host)
    for name in ("../episode_00000.npz", "missing.npz", str(tmp_path / "meta.json")):
        with pytest.raises(ValueError, match="unknown episode"):
            host.load_episode(name)

    host.start_recording()
    with pytest.raises(ValueError, match="stop recording"):
        host.load_episode("episode_00000.npz")
    host.stop_recording(None)

    host._recorder.load_states = lambda file: (np.zeros((3, 4)), {"file": file})
    with pytest.raises(ValueError, match="different model"):
        host.load_episode("episode_00000.npz")


def test_playback_drops_contact_force_because_it_cannot_be_reproduced(tmp_path):
    host = make_host(tmp_path)
    for _ in range(300):
        host._tick_single()  # let the box settle onto the floor
    host._publish()
    live = host.latest_state()["gripper_contacts"]
    assert live and live[0]["force_n"] > 0
    record_episode(host)

    host.load_episode("episode_00000.npz")

    contacts = host.latest_state()["gripper_contacts"]
    assert contacts and all(c["force_n"] is None for c in contacts)
