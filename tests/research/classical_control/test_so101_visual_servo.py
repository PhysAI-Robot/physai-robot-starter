import numpy as np

from physai.contracts import ImageFrame
from research.classical_control.so101_visual_servo import (
    CameraCalibration,
    ColorBlobDetector,
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


def test_camera_calibration_projects_pixel_to_base_plane():
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


def test_camera_calibration_rejects_plane_behind_camera():
    calibration = CameraCalibration(
        fx=100.0,
        fy=100.0,
        cx=0.0,
        cy=0.0,
        rotation_base_camera=np.eye(3),
        translation_base_camera=[0.0, 0.0, -1.0],
    )

    try:
        calibration.pixel_to_plane([0.0, 0.0], plane_z=-2.0)
    except ValueError as exc:
        assert "behind" in str(exc)
    else:
        raise AssertionError("expected a plane-behind-camera error")
