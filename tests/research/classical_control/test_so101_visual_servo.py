import numpy as np
import pytest

from physai.contracts import ImageFrame
from research.classical_control.so101_visual_servo import (
    CameraCalibration,
    ColorBlobDetector,
    SO101VisualServoPolicy,
    VisualFeature,
    VisualServoPhase,
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


def _bare_policy() -> SO101VisualServoPolicy:
    """A policy with only the attributes `debug_frames`/`debug_camera_names`
    need, bypassing `__init__`'s env/kinematics wiring which is irrelevant
    to these two duck-typed hook methods."""
    policy = object.__new__(SO101VisualServoPolicy)
    policy.camera = "front"
    policy.final_camera = "wrist"
    policy._last_detections = {}
    return policy


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


def test_project_is_the_inverse_of_pixel_to_plane_and_rejects_a_point_behind():
    calibration = CameraCalibration(
        fx=120.0,
        fy=120.0,
        cx=80.0,
        cy=60.0,
        rotation_base_camera=np.array(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]
        ),
        translation_base_camera=[0.1, 0.2, 0.5],
    )

    point = calibration.pixel_to_plane([95.0, 70.0], plane_z=0.0)

    np.testing.assert_allclose(calibration.project(point), [95.0, 70.0])
    with pytest.raises(ValueError, match="behind"):
        calibration.project([0.1, 0.2, 1.0])


class _Stub:
    def __init__(self, **attributes) -> None:
        self.__dict__.update(attributes)


def _grasp_policy(blob_pixel, *, fraction=0.25) -> SO101VisualServoPolicy:
    """A policy whose wrist camera expects the pinch point at pixel (50, 40)
    and whose detector finds a blob at `blob_pixel` (None for no blob)."""
    calibration = CameraCalibration(
        fx=100.0,
        fy=100.0,
        cx=50.0,
        cy=40.0,
        rotation_base_camera=np.eye(3),
        translation_base_camera=[0.0, 0.0, 0.0],
    )
    policy = _bare_policy()
    policy.grasp_check_fraction = fraction
    policy.env = _Stub(pinch_center=lambda: np.array([0.0, 0.0, 1.0]))
    policy._calibration_from_observation = lambda observation, camera: calibration
    feature = (
        None
        if blob_pixel is None
        else VisualFeature(
            pixel=np.asarray(blob_pixel, dtype=float), area=50, confidence=1.0
        )
    )
    policy.detector = _Stub(detect=lambda frame: feature)
    return policy


def _wrist_observation(height=80):
    frame = ImageFrame(
        data=np.zeros((height, 100, 3), dtype=np.uint8), camera_name="wrist"
    )
    return _Stub(images={"wrist": frame})


@pytest.mark.parametrize(
    ("blob", "held"),
    [
        ((55.0, 45.0), True),
        ((50.0 + 15.0, 40.0), True),
        ((50.0, 40.0 + 30.0), False),
        (None, False),
    ],
)
def test_the_grasp_is_held_when_the_blob_is_near_the_projected_pinch_point(blob, held):
    # 80 px high image, 0.25 fraction: the limit is 20 px.
    assert _grasp_policy(blob)._grasp_held(_wrist_observation()) is held


def test_the_grasp_check_does_nothing_when_off_or_without_a_wrist_frame():
    far = (95.0, 70.0)
    assert _grasp_policy(far, fraction=0.0)._grasp_held(_wrist_observation()) is True
    assert _grasp_policy(far)._grasp_held(_Stub(images={})) is True


def test_a_missed_grasp_retries_from_approach_and_then_gives_up():
    policy = _bare_policy()
    policy.max_grasp_retries = 2
    policy._grasp_retries = 0
    policy._failure_reason = None
    policy._target_xy = np.array([0.2, 0.0])
    policy._phase = VisualServoPhase.CLOSE
    policy._phase_steps = policy._settle_steps = policy._stall_steps = 7
    policy._elapsed_steps = 150
    policy._attempt_start_step = 0
    policy.env = _Stub(cfg=_Stub(max_steps=1000))
    policy._refine_from_final_camera = lambda observation: False

    for expected_retries in (1, 2):
        policy._miss_grasp(None)
        assert policy._phase is VisualServoPhase.APPROACH
        assert policy._target_xy is None
        assert policy._grasp_retries == expected_retries
        assert policy._failure_reason is None
        policy._target_xy = np.array([0.2, 0.0])
        policy._phase = VisualServoPhase.CLOSE

    policy._miss_grasp(None)
    assert policy._phase is VisualServoPhase.DONE
    assert policy._failure_reason == "grasp_missed"


def test_the_lift_check_compares_the_blob_with_where_it_was_after_closing():
    # Wrist camera expects the pinch point at (50, 40); 80 px high, 0.04 -> 3.2 px.
    policy = _grasp_policy((60.0, 40.0))
    policy.lift_check_fraction = 0.04
    observation = _wrist_observation()
    assert policy._grasp_held(observation) is True  # 10 px off, within 20 px

    assert policy._still_held(observation) is True  # blob has not moved
    policy.detector = _Stub(
        detect=lambda frame: VisualFeature(
            pixel=np.array([60.0, 48.0]), area=50, confidence=1.0
        )
    )
    assert policy._still_held(observation) is False  # moved 8 px
    policy.detector = _Stub(detect=lambda frame: None)
    assert policy._still_held(observation) is False  # gone


def test_the_lift_check_does_nothing_without_a_close_reading_or_when_off():
    policy = _grasp_policy((60.0, 40.0))
    policy.lift_check_fraction = 0.04
    policy._close_offset = None
    assert policy._still_held(_wrist_observation()) is True
    policy._close_offset = np.array([0.0, 0.0])
    policy.lift_check_fraction = 0.0
    assert policy._still_held(_wrist_observation()) is True


def test_a_retry_aims_at_the_wrist_estimate_and_gives_up_when_no_attempt_fits():
    policy = _bare_policy()
    policy.max_grasp_retries = 2
    policy._grasp_retries = 0
    policy._failure_reason = None
    policy._phase = VisualServoPhase.CLOSE
    policy._phase_steps = policy._settle_steps = policy._stall_steps = 0
    policy._elapsed_steps = 150
    policy._attempt_start_step = 0
    policy.env = _Stub(cfg=_Stub(max_steps=1000))

    target = np.array([0.25, 0.05])
    policy._target_xy = np.array([0.2, 0.0])

    def aim(observation):
        policy._target_xy = target
        return True

    policy._refine_from_final_camera = aim
    policy._miss_grasp(None)
    # The wrist estimate replaces the target; nothing is re-detected from the front.
    np.testing.assert_allclose(policy._target_xy, target)
    assert policy._phase is VisualServoPhase.APPROACH
    assert policy._attempt_start_step == 150

    # 200 elapsed, the last attempt took 50, and 80 more are needed: 330 > 300.
    policy.env = _Stub(cfg=_Stub(max_steps=300))
    policy._elapsed_steps = 300 - 100
    policy._phase = VisualServoPhase.CLOSE
    policy._miss_grasp(None)
    assert policy._phase is VisualServoPhase.DONE
    assert policy._failure_reason == "grasp_missed"
    assert policy._grasp_retries == 1


def test_target_disc_detector_finds_the_green_disc_and_ignores_the_rest():
    from research.classical_control.so101_visual_servo import TargetDiscDetector

    image = np.full((60, 80, 3), (190, 200, 195), dtype=np.uint8)
    image[20:30, 40:50] = (68, 184, 100)
    image[5:15, 5:15] = (220, 60, 45)
    detector = TargetDiscDetector()
    feature = detector.detect(image)
    assert feature is not None
    np.testing.assert_allclose(feature.pixel, [44.5, 24.5])
    assert detector.detect(image[:18, :30]) is None
