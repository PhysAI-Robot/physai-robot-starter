import numpy as np
import pytest

from physai.contracts import ImageFrame
from research.classical_control.so101_visual_servo import (
    CameraCalibration,
    ColorBlobDetector,
    SO101VisualServoPolicy,
    VisualFeature,
    draw_crosshair,
)


def test_color_blob_detector_returns_weighted_centroid():
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[7:12, 10:16] = [220, 60, 45]

    feature = ColorBlobDetector(min_area=4).detect(
        ImageFrame(data=image, camera_name="front")
    )

    assert feature is not None
    np.testing.assert_allclose(feature.pixel, [12.5, 9.0])
    assert feature.area == 30
    assert feature.confidence > 0.99


def test_camera_calibration_projects_pixels_and_rejects_a_plane_behind_it():
    calibration = CameraCalibration(
        fx=100.0,
        fy=100.0,
        cx=50.0,
        cy=50.0,
        rotation_base_camera=np.eye(3),
        translation_base_camera=[0.0, 0.0, -1.0],
    )

    point = calibration.pixel_to_plane([60.0, 40.0], plane_z=0.0)

    np.testing.assert_allclose(point, [0.1, -0.1, 0.0])
    with pytest.raises(ValueError, match="behind"):
        calibration.pixel_to_plane([50.0, 50.0], plane_z=-2.0)


def test_draw_crosshair_marks_the_pixel_and_clips_at_the_border():
    image = np.zeros((20, 20, 3), dtype=np.uint8)

    annotated = draw_crosshair(image, np.array([10.0, 8.0]), size=3, color=(0, 255, 0))

    assert annotated is not image
    np.testing.assert_array_equal(image, np.zeros((20, 20, 3), dtype=np.uint8))
    assert tuple(annotated[8, 10]) == (0, 255, 0)
    assert tuple(annotated[8, 7]) == (0, 255, 0)
    assert tuple(annotated[0, 0]) == (0, 0, 0)

    small = np.zeros((10, 10, 3), dtype=np.uint8)
    assert draw_crosshair(small, np.array([-5.0, 4.0])).shape == small.shape


def _bare_policy() -> SO101VisualServoPolicy:
    """A policy with only the attributes `debug_frames`/`debug_camera_names`
    need, bypassing `__init__`'s env/kinematics wiring which is irrelevant
    to these two duck-typed hook methods."""
    policy = object.__new__(SO101VisualServoPolicy)
    policy.camera = "front"
    policy.final_camera = "wrist"
    policy._last_detections = {}
    return policy


def test_debug_frames_are_empty_until_a_detection_then_one_overlay_per_camera():
    policy = _bare_policy()

    assert policy.debug_camera_names == ("front:detections", "wrist:detections")
    assert policy.debug_frames() == {}

    frame = np.zeros((6, 6, 3), dtype=np.uint8)
    feature = VisualFeature(pixel=np.array([2.0, 3.0]), area=5, confidence=0.9)
    policy._last_detections["front"] = (frame, feature)

    frames = policy.debug_frames()

    assert set(frames) == {"front:detections"}
    assert frames["front:detections"].shape == frame.shape
