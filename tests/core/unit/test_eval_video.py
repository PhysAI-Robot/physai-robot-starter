import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import _video  # noqa: E402

NAME = _video.default_video_name("mujoco", "so101", "visual_servo")


@pytest.mark.parametrize(
    ("mode", "success", "kept"),
    [
        (None, True, False),
        (None, False, False),
        ("all", True, True),
        ("all", False, True),
        ("failures", True, False),
        ("failures", False, True),
        ("successes", True, True),
        ("successes", False, False),
    ],
)
def test_keep_video_follows_the_mode_and_the_outcome(mode, success, kept):
    assert _video.keep_video(mode, success) is kept


def test_next_video_stem_is_plain_first_then_counts_up_from_02(tmp_path):
    first = tmp_path / "mujoco_so101_visual_servo_seed0101"
    assert _video.next_video_stem(tmp_path, NAME, 101) == first

    first.with_suffix(".mp4").touch()
    assert (
        _video.next_video_stem(tmp_path, NAME, 101)
        == tmp_path / "mujoco_so101_visual_servo_seed0101_02"
    )

    (tmp_path / "mujoco_so101_visual_servo_seed0101_02.gif").touch()
    # Other seeds, policies and unrelated names do not count.
    (tmp_path / "mujoco_so101_visual_servo_seed0102_09.mp4").touch()
    (tmp_path / "mujoco_so101_scripted_seed0101_07.mp4").touch()
    (tmp_path / "mujoco_so101_visual_servo_seed0101_notes.txt").touch()
    assert (
        _video.next_video_stem(tmp_path, NAME, 101)
        == tmp_path / "mujoco_so101_visual_servo_seed0101_03"
    )

    # Past 99 repeats the number just grows a digit instead of failing.
    (tmp_path / "mujoco_so101_visual_servo_seed0101_99.mp4").touch()
    assert (
        _video.next_video_stem(tmp_path, NAME, 101)
        == tmp_path / "mujoco_so101_visual_servo_seed0101_100"
    )


def test_the_default_name_starts_with_the_simulator_and_an_override_replaces_it(
    tmp_path,
):
    assert NAME == "mujoco_so101_visual_servo"
    assert _video.next_video_stem(tmp_path, "my_run", 7) == tmp_path / "my_run_seed0007"
    (tmp_path / "my_run_seed0007.mp4").touch()
    assert (
        _video.next_video_stem(tmp_path, "my_run", 7) == tmp_path / "my_run_seed0007_02"
    )


def test_a_video_stem_ends_with_its_outcome_and_each_outcome_counts_alone(tmp_path):
    ok = tmp_path / "mujoco_so101_visual_servo_seed0101_o"
    assert _video.next_video_stem(tmp_path, NAME, 101, outcome="_o") == ok
    ok.with_suffix(".mp4").touch()
    assert (
        _video.next_video_stem(tmp_path, NAME, 101, outcome="_o")
        == tmp_path / "mujoco_so101_visual_servo_seed0101_o_02"
    )
    # A failure of the same seed starts its own count; a recording stem is unaffected.
    assert (
        _video.next_video_stem(tmp_path, NAME, 101, outcome="_x")
        == tmp_path / "mujoco_so101_visual_servo_seed0101_x"
    )
    assert _video.next_video_stem(tmp_path, NAME, 101, (".npz",)) == (
        tmp_path / "mujoco_so101_visual_servo_seed0101"
    )


def test_a_view_camera_looks_at_its_target_from_the_given_side():
    import numpy as np
    from physai.sim.workspace import ViewCamera

    view = ViewCamera(lookat=(0.2, 0.0, 0.1), distance=2.0, azimuth=135, elevation=-30)
    position, xyaxes = view.pose()
    right, up = xyaxes[:3], xyaxes[3:]
    forward = np.cross(up, right)  # MuJoCo cameras look along -z = x cross y
    assert np.linalg.norm(position - view.lookat) == pytest.approx(2.0)
    assert forward == pytest.approx((np.asarray(view.lookat) - position) / 2.0)
    assert position[0] > 0.2 and position[1] < 0.0 and position[2] > 0.1
    assert right @ up == pytest.approx(0.0) and up[2] > 0


def test_the_video_observer_prefers_the_view_and_falls_back_to_a_camera():
    from types import SimpleNamespace

    from physai.sim.workspace import VIDEO_CAMERA, WorkspaceConfig

    spec = SimpleNamespace(camera_frames={"front": "camera_front"})
    robot = SimpleNamespace(
        cfg=SimpleNamespace(scene=WorkspaceConfig()), render_camera=lambda name: [name]
    )
    runtime = SimpleNamespace(robot=robot, robot_spec=spec)
    assert _video.VideoObserver(runtime).camera == VIDEO_CAMERA
    assert _video.VideoObserver(runtime, "front").camera == "front"

    bare = SimpleNamespace(robot=SimpleNamespace(), robot_spec=spec)
    assert _video.VideoObserver(bare).camera == "front"
    with pytest.raises(ValueError, match="no `video` view"):
        _video.VideoObserver(bare, VIDEO_CAMERA)
