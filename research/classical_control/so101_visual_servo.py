"""Deterministic image-based visual servoing for the SO-101.

The baseline assumes a calibrated, fixed camera. Camera coordinates use the
usual pinhole convention (x right, y down, z forward); ``rotation_base_camera``
maps that frame into the robot base frame and ``translation_base_camera`` is
the camera origin in the base frame. The wrist camera is moving, so callers
must update its calibration from TF before using it for metric control.

Research module: registers itself with ``physai.robots.registry`` on import.
Core never imports this module directly (see research/README.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np

from physai.contracts import Action, GripperCommand, ImageFrame, Observation
from physai.control.resolver import JointRateLimiter, TwistToJointResolver
from physai.policy.base import Policy
from physai.robots.registry import register_robot_policy

# Normalized gripper aperture commanded while carrying the cube. It must sit
# past the aperture where the fingertip pads first touch the 28 mm cube
# (about 0.176, with the pads fitted to the fingertips) so the position
# controller keeps squeezing at the torque cap, yet within 0.06 of where the
# jaws actually stop or the CLOSE phase never counts as settled. Tuned to the
# old, proud pads as 0.19; 0.13-0.17 all work with the current ones.
SQUEEZE_GRIP = 0.15

# Per-control-step gripper displacement below which the jaw is treated as
# physically stalled (blocked by the object) rather than still closing, used
# by the CLOSE/RELEASE settle condition below. MuJoCo's own slowest,
# still-converging closing rate during a normal CLOSE is about 0.006 per
# step (measured via research/classical_control/FINDINGS.md's trace); a
# gripper actuator that is genuinely jammed against an object (seen on
# Isaac's PhysX position drive, which otherwise tracks its ramping command
# with ~no lag) stops moving almost entirely (~0 per step). This sits below
# the former and above simulator noise.
_GRIP_STALL_EPS = 0.003

# Consecutive stalled steps required before CLOSE/RELEASE accepts a stall as
# final settlement (much longer than `tracking`'s 8-step patience -- see the
# comment in `act()`): long enough to outlast the stick-slip pause measured
# against real Isaac Sim (about 15 steps stuck before it let go the rest of
# the way), short enough to stay well under the 120-step phase cap for a
# jaw that is genuinely, permanently blocked.
_GRIP_STALL_SETTLE_STEPS = 30

# Steps a retry still needs after its grasp, to lift, transfer, lower and release.
_RETRY_TRANSFER_STEPS = 80


@dataclass(frozen=True)
class VisualFeature:
    """A detected image feature in pixel coordinates."""

    pixel: np.ndarray
    area: int
    confidence: float

    def __post_init__(self) -> None:
        pixel = np.asarray(self.pixel, dtype=np.float64).reshape(2)
        if not np.isfinite(pixel).all() or self.area <= 0:
            raise ValueError(
                "visual feature must have a finite pixel and positive area"
            )
        object.__setattr__(self, "pixel", pixel)


class ColorBlobDetector:
    """Detect the dominant red blob by saturation and hue, so lighting does not move it.

    A pixel belongs to the blob when it is red-dominant, saturated
    (`(r - max(g, b)) / r >= min_saturation`) and red rather than orange or
    yellow (`(g - b) / (r - b) <= max_hue_ratio`: ~0 for red and pink, ~0.8
    for the yellow arm). Saturation is unchanged by how brightly a face is
    lit, so the lit, washed-out and shadowed faces of an object are one blob.
    Thresholding on distance to one bright red (or on absolute chroma)
    dropped the shadowed or the washed-out part, and how much that is differs
    between renderers, so the centroid, and the position triangulated from
    it, shifted by centimetres between MuJoCo and Isaac Sim. The hue test is
    what lets the saturation threshold be low enough to keep a pale-pink
    top face without admitting the arm; the table and floor have none.
    The centroid is the plain mean of the blob's pixels (weighting by
    brightness would pull it toward the lit side again); `target_rgb` only
    sets the saturation that counts as full confidence.
    """

    def __init__(
        self,
        target_rgb: tuple[int, int, int] = (220, 60, 45),
        min_saturation: float = 0.25,
        max_hue_ratio: float = 0.35,
        min_chroma: float = 15.0,
        min_value: float = 30.0,
        min_area: int = 8,
    ) -> None:
        self.target_rgb = np.asarray(target_rgb, dtype=np.float64)
        self.min_saturation = float(min_saturation)
        self.max_hue_ratio = float(max_hue_ratio)
        self.min_chroma = float(min_chroma)
        self.min_value = float(min_value)
        self.min_area = int(min_area)
        if (
            self.target_rgb.shape != (3,)
            or not 0 < self.min_saturation < 1
            or not 0 <= self.max_hue_ratio < 1
            or self.min_chroma < 0
            or self.min_value < 1
            or self.min_area < 1
        ):
            raise ValueError("invalid colour detector configuration")
        red = self.target_rgb[0]
        self.target_saturation = float(
            (red - max(self.target_rgb[1], self.target_rgb[2])) / max(red, 1.0)
        )
        if self.target_saturation < self.min_saturation:
            raise ValueError("target_rgb is not saturated enough for min_saturation")

    def detect(self, image: ImageFrame | np.ndarray) -> VisualFeature | None:
        pixels = image.data if isinstance(image, ImageFrame) else np.asarray(image)
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
            raise ValueError("visual detector expects an RGB uint8 image")
        rgb = pixels.astype(np.float64)
        chroma = rgb[..., 0] - np.maximum(rgb[..., 1], rgb[..., 2])
        saturation = chroma / np.maximum(rgb[..., 0], 1.0)
        # 0 for red and pink, towards 1 for orange and yellow.
        hue_ratio = (rgb[..., 1] - rgb[..., 2]) / np.maximum(
            rgb[..., 0] - rgb[..., 2], 1.0
        )
        mask = (
            (saturation >= self.min_saturation)
            & (hue_ratio <= self.max_hue_ratio)
            & (chroma >= self.min_chroma)
            & (rgb[..., 0] >= self.min_value)
        )
        ys, xs = np.nonzero(mask)
        if len(xs) < self.min_area:
            return None
        pixel = np.array([xs.mean(), ys.mean()])
        confidence = float(
            np.clip(saturation[ys, xs].mean() / self.target_saturation, 0.0, 1.0)
        )
        return VisualFeature(pixel=pixel, area=len(xs), confidence=confidence)


class TargetDiscDetector:
    """Detect the green place-target disc by how far green stands above red and blue.

    The disc is the only large green object on the table: its pixels sit at
    26 or more above the larger of red and blue (median 88), while the
    background stays at 12 or less on MuJoCo and reaches 21 on Isaac Sim (the
    table's far edge and the arm), so 30 keeps the disc and drops both. At 20
    those strays pulled Isaac's centroid 10-43 mm off. A cube resting on the
    disc hides part of it and biases the centroid, which is why a task keeps
    them a minimum distance apart.
    """

    def __init__(self, min_chroma: float = 30.0, min_area: int = 20) -> None:
        if min_chroma <= 0 or min_area < 1:
            raise ValueError("invalid target disc detector configuration")
        self.min_chroma = float(min_chroma)
        self.min_area = int(min_area)

    def detect(self, image: ImageFrame | np.ndarray) -> VisualFeature | None:
        pixels = image.data if isinstance(image, ImageFrame) else np.asarray(image)
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
            raise ValueError("visual detector expects an RGB uint8 image")
        rgb = pixels.astype(np.float64)
        chroma = rgb[..., 1] - np.maximum(rgb[..., 0], rgb[..., 2])
        ys, xs = np.nonzero(chroma >= self.min_chroma)
        if len(xs) < self.min_area:
            return None
        return VisualFeature(
            pixel=np.array([xs.mean(), ys.mean()]),
            area=len(xs),
            confidence=float(np.clip(chroma[ys, xs].mean() / 90.0, 0.0, 1.0)),
        )


def draw_crosshair(
    image: np.ndarray,
    pixel: np.ndarray,
    *,
    size: int = 10,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Return a copy of `image` (H, W, 3) uint8 with a crosshair at `pixel` (x, y)."""
    annotated = np.array(image, dtype=np.uint8, copy=True)
    height, width = annotated.shape[:2]
    x, y = int(round(float(pixel[0]))), int(round(float(pixel[1])))
    if 0 <= y < height:
        x0, x1 = max(0, x - size), min(width, x + size + 1)
        annotated[max(0, y - 1) : min(height, y + 2), x0:x1] = color
    if 0 <= x < width:
        y0, y1 = max(0, y - size), min(height, y + size + 1)
        annotated[y0:y1, max(0, x - 1) : min(width, x + 2)] = color
    return annotated


@dataclass(frozen=True)
class CameraCalibration:
    """Pinhole intrinsics and a camera-to-base rigid transform."""

    fx: float
    fy: float
    cx: float
    cy: float
    rotation_base_camera: np.ndarray
    translation_base_camera: np.ndarray

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation_base_camera, dtype=np.float64).reshape(3, 3)
        translation = np.asarray(
            self.translation_base_camera, dtype=np.float64
        ).reshape(3)
        if (
            min(self.fx, self.fy) <= 0
            or not np.isfinite(rotation).all()
            or not np.isfinite(translation).all()
        ):
            raise ValueError(
                "camera calibration must be finite with positive focal lengths"
            )
        object.__setattr__(self, "rotation_base_camera", rotation)
        object.__setattr__(self, "translation_base_camera", translation)

    def project(self, point_base: np.ndarray) -> np.ndarray:
        """The pixel a point given in the base frame appears at (the inverse of
        `ray_base`); raises `ValueError` for a point behind the camera."""
        point = np.asarray(point_base, dtype=np.float64).reshape(3)
        in_camera = self.rotation_base_camera.T @ (point - self.translation_base_camera)
        if in_camera[2] <= 1e-6:
            raise ValueError("point is behind the camera")
        return np.array(
            [
                self.fx * in_camera[0] / in_camera[2] + self.cx,
                self.fy * in_camera[1] / in_camera[2] + self.cy,
            ]
        )

    def ray_base(self, pixel: np.ndarray) -> np.ndarray:
        pixel = np.asarray(pixel, dtype=np.float64).reshape(2)
        ray_camera = np.array(
            [(pixel[0] - self.cx) / self.fx, (pixel[1] - self.cy) / self.fy, 1.0]
        )
        return self.rotation_base_camera @ (ray_camera / np.linalg.norm(ray_camera))

    def pixel_to_plane(self, pixel: np.ndarray, plane_z: float) -> np.ndarray:
        ray = self.ray_base(pixel)
        denominator = ray[2]
        if abs(denominator) < 1e-9:
            raise ValueError("camera ray is parallel to the requested plane")
        distance = (float(plane_z) - self.translation_base_camera[2]) / denominator
        if distance <= 0:
            raise ValueError("requested plane is behind the camera")
        return self.translation_base_camera + distance * ray


@dataclass(frozen=True)
class VisualServoMetrics:
    visual_error_px: float | None = None
    ee_error_m: float | None = None
    settled: bool = False
    failure_reason: str | None = None
    phase: str = "APPROACH"
    phase_steps: int = 0
    settling_time_s: float | None = None
    grasp_retries: int = 0
    feature_px: tuple[float, float] | None = None
    feature_confidence: float | None = None


class VisualServoPhase(Enum):
    APPROACH = auto()
    DESCEND = auto()
    CLOSE = auto()
    LIFT = auto()
    TRANSFER = auto()
    LOWER = auto()
    RELEASE = auto()
    RETREAT = auto()
    DONE = auto()


class SO101VisualServoPolicy(Policy):
    """Detect a cube visually, then execute a bounded pick-and-place motion."""

    name = "visual_servo"

    def __init__(
        self,
        env,
        *,
        camera: str = "front",
        detector: ColorBlobDetector | None = None,
        target_detector: TargetDiscDetector | None = None,
        place_from_camera: bool = True,
        calibration: CameraCalibration | None = None,
        target_pixel: tuple[float, float] | None = None,
        target_plane_z: float = 0.035,
        squeeze_grip: float = SQUEEZE_GRIP,
        kp: float = 2.0,
        pixel_tolerance: float = 8.0,
        ee_tolerance: float = 0.012,
        max_joint_rate: float = 1.2,
        dt: float | None = None,
        final_camera: str = "wrist",
        grasp_offset_xy: tuple[float, float] = (-0.006, -0.006),
        grasp_check_fraction: float = 0.25,
        lift_check_fraction: float = 0.04,
        max_grasp_retries: int = 2,
    ) -> None:
        self.env = env
        self.camera = camera
        self.final_camera = final_camera
        self.detector = detector or ColorBlobDetector()
        # Where to place: the green disc seen by the same camera frame that
        # finds the cube. False reads the simulator's `target_pos` instead
        # (privileged), for a fixed target or to compare against it.
        self.target_detector = target_detector or TargetDiscDetector()
        self.place_from_camera = bool(place_from_camera)
        self._place_xy: np.ndarray | None = None
        self.calibration = calibration
        self.target_pixel = target_pixel
        self.target_plane_z = float(target_plane_z)
        # How far normalized 0 (closed) to 1 (open) `_waypoint()` squeezes
        # during CLOSE/LIFT/TRANSFER/LOWER. Defaults to `SQUEEZE_GRIP`, the
        # value tuned against MuJoCo's contact solver; a stiffer engine (or
        # a different cube/pad friction setup) may need a deeper squeeze to
        # actually hold the object against its own weight before slipping.
        self.squeeze_grip = float(squeeze_grip)
        # Added to the estimated object xy for the pick phases (APPROACH
        # through LIFT). The static jaw leaves ~1 mm of clearance beside a
        # 28 mm cube, so a few mm of estimation bias toward it lands the jaw
        # on top of the object. With correct intrinsics (fx == fy) the blob
        # centroid sits a few mm toward +y/+x of the cube in both simulators;
        # biasing the target 6 mm toward -x/-y keeps the jaws on the safe
        # side. Measured: 20/20 in MuJoCo at 320x240, 640x480 and 1280x720.
        self.grasp_offset_xy = np.asarray(grasp_offset_xy, dtype=np.float64)
        # At the end of CLOSE a cube held between the jaws appears in the wrist
        # camera where the pinch point projects (the pinch point comes from the
        # joint angles, not from the simulator). A blob farther from it than
        # this fraction of the image height, or none at all, means the grasp
        # missed: the policy opens the gripper and starts over, up to
        # `max_grasp_retries` times, then stops with `failure_reason`
        # "grasp_missed". Measured at 320x240 (MuJoCo): a held cube is 10 to 25
        # px away, a missed one 138 to 174 px. 0 turns the check off.
        self.grasp_check_fraction = float(grasp_check_fraction)
        # A cube that is really held does not move in the wrist image while the
        # arm lifts, because it moves with the camera. At the end of LIFT the
        # blob must be within this fraction of the image height of where it was
        # at the end of CLOSE, else the grasp let go (a cube left on the table
        # moved 16 and 23 px, a held one 0.7 px at 240 px high). 0 turns it off.
        self.lift_check_fraction = float(lift_check_fraction)
        self._close_offset: np.ndarray | None = None
        self._attempt_start_step = 0
        self.max_grasp_retries = int(max_grasp_retries)
        self._grasp_retries = 0
        self._failure_reason: str | None = None
        self.kp = float(kp)
        self.pixel_tolerance = float(pixel_tolerance)
        self.ee_tolerance = float(ee_tolerance)
        control_dt = dt or (1.0 / float(getattr(env.cfg, "control_hz", 30.0)))
        self._resolver = TwistToJointResolver(
            env.kin,
            state_provider=lambda: env.data,
            dt=control_dt,
            max_joint_step=0.08,
        )
        self._limiter = JointRateLimiter(max_joint_rate, control_dt)
        self.metrics = VisualServoMetrics()
        self._dt = control_dt
        self._phase = VisualServoPhase.APPROACH
        self._phase_steps = 0
        self._settle_steps = 0
        self._stall_steps = 0
        self._target_xy: np.ndarray | None = None
        self._q_cmd: np.ndarray | None = None
        self._last_feature_px: tuple[float, float] | None = None
        self._last_feature_confidence: float | None = None
        self._grip = 1.0
        self._elapsed_steps = 0
        self._settling_time_s: float | None = None
        self._last_visual_error_px: float | None = None
        self._last_detections: dict[str, tuple[np.ndarray, VisualFeature]] = {}
        self._prev_grip_now: float | None = None

    def _calibration_from_observation(
        self, observation: Observation, camera: str | None = None
    ) -> CameraCalibration:
        """Build a `CameraCalibration` from `ImageFrame.intrinsics`/
        `.extrinsics` (already in the pinhole convention this module uses;
        see its module docstring) instead of reaching into the backend's own
        state, so this policy works against any `RobotPort` that fills those
        fields in, not only a direct MuJoCo environment.
        """
        camera = camera or self.camera
        frame = observation.images.get(camera)
        if frame is None or frame.intrinsics is None or frame.extrinsics is None:
            raise KeyError(f"camera {camera!r} has no calibration in this observation")
        return CameraCalibration(
            fx=frame.intrinsics.fx,
            fy=frame.intrinsics.fy,
            cx=frame.intrinsics.cx,
            cy=frame.intrinsics.cy,
            rotation_base_camera=frame.extrinsics.orientation.to_matrix(),
            translation_base_camera=frame.extrinsics.position.as_array(),
        )

    def reset(self, observation: Observation, goal=None, instruction=None) -> None:
        if self.calibration is None:
            # A host-driven session renders camera frames on a separate
            # worker thread and has not populated any yet at reset time (see
            # `web.host.Host._sync_observation_images`); `act()` retries this
            # once a frame — and its calibration — actually exists.
            try:
                self.calibration = self._calibration_from_observation(observation)
            except KeyError:
                pass
        self._limiter.reset(observation.joint_state.position[:5])
        self.metrics = VisualServoMetrics()
        self._phase = VisualServoPhase.APPROACH
        self._phase_steps = 0
        self._settle_steps = 0
        self._stall_steps = 0
        self._target_xy = None
        self._place_xy = None
        self._q_cmd = observation.joint_state.position[:5].copy()
        self._grip = 1.0
        self._elapsed_steps = 0
        self._settling_time_s = None
        self._last_visual_error_px = None
        self._last_detections = {}
        self._prev_grip_now = None
        self._grasp_retries = 0
        self._failure_reason = None
        self._close_offset = None
        self._attempt_start_step = 0

    def _refine_from_final_camera(self, observation: Observation) -> bool:
        """Re-aim the target at where the wrist camera sees the cube; True when
        it moved (the new estimate is accepted within 8 cm of the old one)."""
        if self._target_xy is None:
            return False
        frame = observation.images.get(self.final_camera)
        if frame is None:
            return False
        feature = self.detector.detect(frame)
        if feature is None:
            return False
        self._last_detections[self.final_camera] = (
            np.asarray(frame.data, dtype=np.uint8).copy(),
            feature,
        )
        target_pixel = (
            self.target_pixel
            or (np.array([frame.width, frame.height], dtype=np.float64) - 1.0) / 2.0
        )
        self._last_visual_error_px = float(np.linalg.norm(feature.pixel - target_pixel))
        try:
            candidate = self._calibration_from_observation(
                observation, self.final_camera
            ).pixel_to_plane(feature.pixel, self.target_plane_z)[:2]
        except (KeyError, ValueError):
            return False
        if np.linalg.norm(candidate - self._target_xy) <= 0.08:
            self._target_xy = candidate
            return True
        return False

    @property
    def done(self) -> bool:
        return self._phase is VisualServoPhase.DONE

    @property
    def debug_camera_names(self) -> tuple[str, ...]:
        """Optional debug-camera hook the web `Host` discovers via duck typing."""
        return (f"{self.camera}:detections", f"{self.final_camera}:detections")

    def debug_frames(self) -> dict[str, np.ndarray]:
        """Optional debug-camera hook: latest detection overlay per camera."""
        return {
            f"{name}:detections": draw_crosshair(frame, feature.pixel)
            for name, (frame, feature) in self._last_detections.items()
        }

    def _solve(self, target: np.ndarray) -> np.ndarray:
        result = self.env.kin.ik_pinch(target, q_init=self._q_cmd)
        return result.qpos if result.converged else self._q_cmd

    def _advance(self) -> None:
        order = list(VisualServoPhase)
        self._phase = order[min(order.index(self._phase) + 1, len(order) - 1)]
        self._phase_steps = 0
        self._settle_steps = 0
        self._stall_steps = 0

    def _blob_offset(self, observation: Observation) -> np.ndarray | None:
        """Where the wrist camera sees the cube, in pixels from the projected
        pinch point (the pinch point comes from the joint angles). `None`
        when no blob is found; `LookupError` when there is no wrist frame or
        calibration to check against."""
        frame = observation.images.get(self.final_camera)
        if frame is None:
            raise LookupError("no wrist frame")
        try:
            calibration = self._calibration_from_observation(
                observation, self.final_camera
            )
            expected = calibration.project(self.env.pinch_center())
        except (KeyError, ValueError) as exc:
            raise LookupError("no wrist calibration") from exc
        feature = self.detector.detect(frame)
        return None if feature is None else feature.pixel - expected

    def _grasp_held(self, observation: Observation) -> bool:
        """Whether the cube is between the jaws at the end of CLOSE.

        Without a wrist frame or calibration there is nothing to check, so the
        grasp is taken as held (the policy then behaves as without the check).
        """
        self._close_offset = None
        if self.grasp_check_fraction <= 0:
            return True
        try:
            offset = self._blob_offset(observation)
        except LookupError:
            return True
        if offset is None:
            return False
        self._close_offset = offset
        limit = self.grasp_check_fraction * observation.images[self.final_camera].height
        return bool(np.linalg.norm(offset) <= limit)

    def _still_held(self, observation: Observation) -> bool:
        """Whether the cube rose with the gripper: at the end of LIFT it must
        still sit where it did at the end of CLOSE in the wrist image."""
        if self.lift_check_fraction <= 0 or self._close_offset is None:
            return True
        try:
            offset = self._blob_offset(observation)
        except LookupError:
            return True
        if offset is None:
            return False
        limit = self.lift_check_fraction * observation.images[self.final_camera].height
        return bool(np.linalg.norm(offset - self._close_offset) <= limit)

    def _miss_grasp(self, observation: Observation) -> None:
        """Open and try again from APPROACH, or give up.

        The retry aims at where the wrist camera sees the cube from the pose
        that just missed (the pinch is then close to it, so the estimate is the
        best the policy has), and only falls back to the front camera when the
        wrist sees nothing usable. It gives up, with `failure_reason`
        "grasp_missed", once the retries are used or when the steps left cannot
        fit another attempt (about what the failed one took plus the
        transfer).
        """
        attempt_steps = self._elapsed_steps - self._attempt_start_step
        limit = getattr(getattr(self.env, "cfg", None), "max_steps", None)
        fits = limit is None or (
            self._elapsed_steps + attempt_steps + _RETRY_TRANSFER_STEPS <= limit
        )
        if self._grasp_retries < self.max_grasp_retries and fits:
            self._grasp_retries += 1
            if not self._refine_from_final_camera(observation):
                self._target_xy = None
            self._phase = VisualServoPhase.APPROACH
            self._phase_steps = 0
            self._settle_steps = 0
            self._stall_steps = 0
            self._attempt_start_step = self._elapsed_steps
        else:
            self._failure_reason = "grasp_missed"
            self._phase = VisualServoPhase.DONE
            self._phase_steps = 0

    def _waypoint(self) -> tuple[np.ndarray, float]:
        if self._target_xy is None:
            raise RuntimeError("visual target is not initialized")
        rest_z = self.env.rest_z
        cube_z = self.target_plane_z
        pick_xy = self._target_xy + self.grasp_offset_xy
        if self._phase is VisualServoPhase.APPROACH:
            return np.array([*pick_xy, cube_z + 0.045]), 1.0
        if self._phase in (VisualServoPhase.DESCEND, VisualServoPhase.CLOSE):
            return np.array(
                [*pick_xy, cube_z]
            ), 1.0 if self._phase is VisualServoPhase.DESCEND else self.squeeze_grip
        if self._phase is VisualServoPhase.LIFT:
            return np.array([*pick_xy, rest_z + 0.035]), self.squeeze_grip
        place_x, place_y = self._place()
        if self._phase is VisualServoPhase.TRANSFER:
            return np.array([place_x, place_y, rest_z + 0.035]), self.squeeze_grip
        if self._phase in (VisualServoPhase.LOWER, VisualServoPhase.RELEASE):
            return np.array(
                [place_x, place_y, rest_z + 0.016]
            ), self.squeeze_grip if self._phase is VisualServoPhase.LOWER else 1.0
        return np.array([place_x, place_y, rest_z + 0.035]), 1.0

    def _place(self) -> np.ndarray:
        if not self.place_from_camera:
            return self.env.target_pos[:2]
        if self._place_xy is None:
            raise RuntimeError("place target is not initialized")
        return self._place_xy

    def act(self, observation: Observation) -> Action:
        if self._target_xy is None:
            frame = observation.images.get(self.camera)
            if frame is None:
                self.metrics = VisualServoMetrics(
                    failure_reason=f"missing_camera:{self.camera}"
                )
                return Action(
                    joint_position=self._q_cmd, gripper=GripperCommand(position=1.0)
                )
            if self.calibration is None:
                # `reset()` deferred this: the first frame(s) after a
                # host-driven reset can lag the camera worker.
                try:
                    self.calibration = self._calibration_from_observation(observation)
                except KeyError:
                    self.metrics = VisualServoMetrics(
                        failure_reason=f"missing_camera:{self.camera}"
                    )
                    return Action(
                        joint_position=self._q_cmd, gripper=GripperCommand(position=1.0)
                    )
            feature = self.detector.detect(frame)
            if feature is None:
                self.metrics = VisualServoMetrics(failure_reason="feature_not_found")
                return Action(
                    joint_position=self._q_cmd, gripper=GripperCommand(position=1.0)
                )
            self._last_feature_px = tuple(float(v) for v in feature.pixel)
            self._last_feature_confidence = float(feature.confidence)
            self._last_detections[self.camera] = (
                np.asarray(frame.data, dtype=np.uint8).copy(),
                feature,
            )
            target_pixel = (
                self.target_pixel
                or (np.array([frame.width, frame.height], dtype=np.float64) - 1.0) / 2.0
            )
            self._last_visual_error_px = float(
                np.linalg.norm(feature.pixel - target_pixel)
            )
            try:
                self._target_xy = self.calibration.pixel_to_plane(
                    feature.pixel, self.target_plane_z
                )[:2]
            except ValueError as exc:
                self.metrics = VisualServoMetrics(failure_reason=str(exc))
                return Action(
                    joint_position=self._q_cmd, gripper=GripperCommand(position=1.0)
                )
            if self.place_from_camera and self._place_xy is None:
                disc = self.target_detector.detect(frame)
                if disc is None:
                    self._target_xy = None
                    self.metrics = VisualServoMetrics(failure_reason="target_not_found")
                    return Action(
                        joint_position=self._q_cmd, gripper=GripperCommand(position=1.0)
                    )
                self._place_xy = self.calibration.pixel_to_plane(
                    disc.pixel, self.env.table_top
                )[:2]

        if self._phase is VisualServoPhase.DESCEND:
            self._refine_from_final_camera(observation)
        target, grip_goal = self._waypoint()
        self._grip = float(
            np.clip(grip_goal, self._grip - 0.9 * self._dt, self._grip + 0.9 * self._dt)
        )
        self._q_cmd = self._limiter(self._solve(target))
        pinch = self.env.pinch_center()
        reached = float(np.linalg.norm(pinch - target)) <= self.ee_tolerance
        self._elapsed_steps += 1
        if reached and self._settling_time_s is None:
            self._settling_time_s = self._elapsed_steps * self._dt
        grip_now = self.env.joint_to_gripper(observation.joint_state.position[-1])
        # A gripper actuator that tracks its ramping command with ~no lag
        # (seen on Isaac's PhysX position drive) satisfies `tracking` a
        # handful of steps after CLOSE/RELEASE starts, long before the ramp
        # itself reaches `grip_goal` -- exiting the squeeze at whatever
        # fraction of the ramp had elapsed by then instead of at the
        # requested depth. `ramp_done` requires the ramp to finish first.
        ramp_done = self._grip == grip_goal
        tracking = ramp_done and abs(grip_now - self._grip) < 0.06
        # A jaw physically resisted by the object (position stopped moving,
        # but still farther than 0.06 from the fully-ramped command) would
        # then never satisfy `tracking` -- `stalled` is the alternative
        # settle path. But a jammed position-controlled gripper can stick on
        # static friction for a while and then suddenly slip several steps
        # later once it overcomes it (confirmed against real Isaac Sim: held
        # at ~0.27 open, 0.21 short of a 0.06 target, for about 15 steps,
        # then closed the rest of the way in 2); treating that mid-slip
        # pause as final settlement exits CLOSE, and therefore starts the
        # next phase's arm motion, before the slip finishes -- the launch
        # this policy's shared grasp choreography otherwise avoids by never
        # moving the arm until the squeeze is done (as CLOSE's own waypoint does).
        # Requiring a much longer stall before accepting it gives a mid-slip
        # pause time to resolve on its own first.
        stalled = ramp_done and (
            self._prev_grip_now is not None
            and abs(grip_now - self._prev_grip_now) < _GRIP_STALL_EPS
        )
        self._prev_grip_now = grip_now
        self._phase_steps += 1
        if self._phase in (VisualServoPhase.CLOSE, VisualServoPhase.RELEASE):
            self._settle_steps = self._settle_steps + 1 if tracking else 0
            self._stall_steps = self._stall_steps + 1 if stalled else 0
            if (
                self._settle_steps >= 8
                or self._stall_steps >= _GRIP_STALL_SETTLE_STEPS
                or self._phase_steps >= 120
            ):
                if self._phase is VisualServoPhase.CLOSE and not self._grasp_held(
                    observation
                ):
                    self._miss_grasp(observation)
                else:
                    self._advance()
        elif reached or self._phase_steps >= 120:
            if self._phase is VisualServoPhase.LIFT and not self._still_held(
                observation
            ):
                self._miss_grasp(observation)
            else:
                self._advance()
        self.metrics = VisualServoMetrics(
            visual_error_px=self._last_visual_error_px,
            ee_error_m=float(np.linalg.norm(pinch - target)),
            settled=reached,
            phase=self._phase.name,
            phase_steps=self._phase_steps,
            settling_time_s=self._settling_time_s,
            grasp_retries=self._grasp_retries,
            failure_reason=self._failure_reason,
            feature_px=self._last_feature_px,
            feature_confidence=self._last_feature_confidence,
        )
        return Action(
            joint_position=self._q_cmd, gripper=GripperCommand(position=self._grip)
        )


def make_visual_servo_policy(
    *, env, cfg: Any = None, **kwargs: Any
) -> SO101VisualServoPolicy:
    """Build the deterministic SO-101 camera-feedback baseline."""
    options = dict(kwargs)
    if cfg is not None:
        options.update(cfg if isinstance(cfg, dict) else vars(cfg))
    return SO101VisualServoPolicy(env, **options)


register_robot_policy("so101", "visual_servo", make_visual_servo_policy)


__all__ = [
    "CameraCalibration",
    "ColorBlobDetector",
    "SO101VisualServoPolicy",
    "TargetDiscDetector",
    "VisualFeature",
    "VisualServoMetrics",
    "draw_crosshair",
    "make_visual_servo_policy",
]
