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


def test_next_video_stem_in_a_missing_directory_is_the_plain_name(tmp_path):
    assert (
        _video.next_video_stem(tmp_path / "missing", "mujoco_so101_scripted", 5)
        == tmp_path / "missing" / "mujoco_so101_scripted_seed0005"
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
