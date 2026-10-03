import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import compare_cameras  # noqa: E402


def image(color, size=(60, 80)) -> np.ndarray:
    return np.full((*size, 3), color, dtype=np.uint8)


def test_the_arm_and_the_cube_are_told_apart_by_colour():
    arm_pixel = image((220, 190, 40))
    cube_pixel = image((200, 60, 50))
    grey = image((180, 180, 180))

    assert compare_cameras.robot_mask(arm_pixel).all()
    assert not compare_cameras.robot_mask(cube_pixel).any()
    assert compare_cameras.cube_mask(cube_pixel).all()
    assert not compare_cameras.cube_mask(arm_pixel).any()
    assert not compare_cameras.cube_mask(grey).any()


def test_region_means_split_cube_background_and_floor():
    frame = image((200, 200, 200))
    frame[:20] = (100, 120, 140)  # upper third (60 rows // 3): background
    frame[40:50, 10:30] = (200, 60, 50)  # a cube in the lower half

    means = compare_cameras.region_means(frame)

    assert np.allclose(means["cube"], (200, 60, 50))
    assert np.allclose(means["background"], (100, 120, 140))
    assert np.allclose(means["table/floor"], (200, 200, 200))


def test_pixel_difference_ignores_the_arm_in_either_image():
    a, b = image((100, 100, 100)), image((120, 100, 100))
    assert compare_cameras.pixel_difference(a, b) == np.mean([20, 0, 0])

    b[:30] = (220, 190, 40)  # the arm covers the top half of b
    assert compare_cameras.pixel_difference(a, b) == np.mean([20, 0, 0])
