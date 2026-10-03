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


def test_color_blob_detector_returns_the_blob_centroid():
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


def test_color_blob_detector_counts_the_shadowed_part_of_an_object():
    """A lit face and a shadowed face of the same cube are one blob: lighting
    must not move the centroid (it is what differs between simulators)."""
    image = np.full((40, 60, 3), 200, dtype=np.uint8)
    image[10:20, 10:30] = [210, 60, 50]  # lit top face
    image[20:30, 10:30] = [95, 28, 22]  # the same cube, in shadow

    feature = ColorBlobDetector().detect(image)

    assert feature is not None
    assert feature.area == 20 * 20
    assert feature.pixel[1] == pytest.approx(19.5, abs=0.7)  # middle of both faces


def test_color_blob_detector_ignores_the_yellow_arm_and_the_table():
    image = np.full((30, 30, 3), (208, 208, 193), dtype=np.uint8)  # table
    image[5:15, 5:15] = [225, 190, 40]  # yellow arm paint
    image[18:26, 18:26] = [100, 80, 10]  # the arm in shadow

    assert ColorBlobDetector().detect(image) is None


def test_color_blob_detector_keeps_a_washed_out_face_but_not_orange():
    """An over-lit face is washed out (saturation ~0.33) yet still red; orange
    paint stays out by hue even at a similar saturation (~0.29)."""
    image = np.full((40, 60, 3), 200, dtype=np.uint8)
    image[5:15, 5:25] = [240, 160, 150]  # washed-out top face: saturation 0.33
    image[15:25, 5:25] = [198, 118, 100]  # the same cube's front face
    image[28:38, 30:50] = [225, 160, 40]  # orange paint: sat ~0.29, hue-ratio ~0.65

    feature = ColorBlobDetector().detect(image)

    assert feature is not None
    assert feature.area == 20 * 20  # pale face and front face, nothing orange
    assert feature.pixel[1] == pytest.approx(14.5, abs=0.2)
