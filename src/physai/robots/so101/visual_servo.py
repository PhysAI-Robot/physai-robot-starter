"""Deterministic image-based visual servoing for the SO-101.

The baseline assumes a calibrated, fixed camera. Camera coordinates use the
usual pinhole convention (x right, y down, z forward); ``rotation_base_camera``
maps that frame into the robot base frame and ``translation_base_camera`` is
the camera origin in the base frame. The wrist camera is moving, so callers
must update its calibration from TF before using it for metric control.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from ...contracts import Action, GripperCommand, ImageFrame, Observation, Twist, Vector3
from ...control.resolver import JointRateLimiter, TwistToJointResolver
from ...policy.base import Policy


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
    """Detect the dominant blob nearest a target RGB colour without OpenCV."""

    def __init__(
        self,
        target_rgb: tuple[int, int, int] = (220, 60, 45),
        tolerance: float = 90.0,
        min_area: int = 8,
    ) -> None:
        self.target_rgb = np.asarray(target_rgb, dtype=np.float64)
        self.tolerance = float(tolerance)
        self.min_area = int(min_area)
        if self.target_rgb.shape != (3,) or self.tolerance <= 0 or self.min_area < 1:
            raise ValueError("invalid colour detector configuration")

    def detect(self, image: ImageFrame | np.ndarray) -> VisualFeature | None:
        pixels = image.data if isinstance(image, ImageFrame) else np.asarray(image)
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
            raise ValueError("visual detector expects an RGB uint8 image")
        distance = np.linalg.norm(pixels.astype(np.float64) - self.target_rgb, axis=2)
        mask = distance <= self.tolerance
        ys, xs = np.nonzero(mask)
        if len(xs) < self.min_area:
            return None
        weights = np.maximum(self.tolerance - distance[ys, xs], 1.0)
        pixel = np.array(
            [np.average(xs, weights=weights), np.average(ys, weights=weights)]
        )
        confidence = float(
            np.clip(1.0 - np.average(distance[ys, xs]) / self.tolerance, 0.0, 1.0)
        )
        return VisualFeature(pixel=pixel, area=len(xs), confidence=confidence)


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
        calibration: CameraCalibration | None = None,
        target_pixel: tuple[float, float] | None = None,
        target_plane_z: float = 0.035,
        kp: float = 2.0,
        max_speed: float = 0.08,
        pixel_tolerance: float = 8.0,
        ee_tolerance: float = 0.012,
        max_joint_rate: float = 1.2,
        dt: float | None = None,
        final_camera: str = "wrist",
    ) -> None:
        self.env = env
        self.camera = camera
        self.final_camera = final_camera
        self.detector = detector or ColorBlobDetector()
        self.calibration = calibration
        self.target_pixel = target_pixel
        self.target_plane_z = float(target_plane_z)
        self.kp = float(kp)
        self.max_speed = float(max_speed)
        self.pixel_tolerance = float(pixel_tolerance)
        self.ee_tolerance = float(ee_tolerance)
        control_dt = dt or (1.0 / float(getattr(env.cfg, "control_hz", 25.0)))
        self._resolver = TwistToJointResolver(
            env.kin, data=env.data, dt=control_dt, max_joint_step=0.08
        )
        self._limiter = JointRateLimiter(max_joint_rate, control_dt)
        self.metrics = VisualServoMetrics()
        self._dt = control_dt
        self._phase = VisualServoPhase.APPROACH
        self._phase_steps = 0
        self._settle_steps = 0
        self._target_xy: np.ndarray | None = None
        self._q_cmd: np.ndarray | None = None
        self._grip = 1.0

    def _calibration_from_env(self, camera: str | None = None) -> CameraCalibration:
        camera = camera or self.camera
        camera_id = self.env.model.camera(camera).id
        if camera_id < 0:
            raise KeyError(f"camera {self.camera!r} missing from model")
        height = self.env.cfg.scene.camera_height
        width = self.env.cfg.scene.camera_width
        fy = height / (
            2.0 * np.tan(np.deg2rad(self.env.model.cam_fovy[camera_id]) / 2.0)
        )
        return CameraCalibration(
            fx=fy * width / height,
            fy=fy,
            cx=(width - 1) / 2.0,
            cy=(height - 1) / 2.0,
            # MuJoCo camera frames use +x right, +y up, and -z forward;
            # the public pinhole frame uses +x right, +y down, and +z forward.
            rotation_base_camera=self.env.data.cam_xmat[camera_id].reshape(3, 3)
            @ np.diag([1.0, -1.0, -1.0]),
            translation_base_camera=self.env.data.cam_xpos[camera_id],
        )

    def reset(self, observation: Observation, goal=None, instruction=None) -> None:
        if self.calibration is None:
            self.calibration = self._calibration_from_env()
        self._limiter.reset(observation.joint_state.position[:5])
        self.metrics = VisualServoMetrics()
        self._phase = VisualServoPhase.APPROACH
        self._phase_steps = 0
        self._settle_steps = 0
        self._target_xy = None
        self._q_cmd = observation.joint_state.position[:5].copy()
        self._grip = 1.0

    def _refine_from_final_camera(self) -> None:
        if self._target_xy is None:
            return
        frame = self.env.observe().images.get(self.final_camera)
        if frame is None:
            return
        feature = self.detector.detect(frame)
        if feature is None:
            return
        try:
            candidate = self._calibration_from_env(self.final_camera).pixel_to_plane(
                feature.pixel, self.target_plane_z
            )[:2]
        except ValueError:
            return
        if np.linalg.norm(candidate - self._target_xy) <= 0.08:
            self._target_xy = candidate

    @property
    def done(self) -> bool:
        return self._phase is VisualServoPhase.DONE

    def _solve(self, target: np.ndarray) -> np.ndarray:
        result = self.env.kin.ik_pinch(target, q_init=self._q_cmd)
        return result.qpos if result.converged else self._q_cmd

    def _advance(self) -> None:
        order = list(VisualServoPhase)
        self._phase = order[min(order.index(self._phase) + 1, len(order) - 1)]
        self._phase_steps = 0
        self._settle_steps = 0

    def _waypoint(self) -> tuple[np.ndarray, float]:
        if self._target_xy is None:
            raise RuntimeError("visual target is not initialized")
        table_top = self.env.cfg.scene.table_pos[2] + self.env.cfg.scene.table_size[2]
        rest_z = table_top + self.env.cfg.scene.cube_half
        cube_z = self.target_plane_z
        if self._phase is VisualServoPhase.APPROACH:
            return np.array([*self._target_xy, cube_z + 0.045]), 1.0
        if self._phase in (VisualServoPhase.DESCEND, VisualServoPhase.CLOSE):
            return np.array(
                [*self._target_xy, cube_z]
            ), 1.0 if self._phase is VisualServoPhase.DESCEND else 0.19
        if self._phase is VisualServoPhase.LIFT:
            return np.array([*self._target_xy, rest_z + 0.035]), 0.19
        if self._phase is VisualServoPhase.TRANSFER:
            return np.array(
                [self.env.target_pos[0], self.env.target_pos[1], rest_z + 0.035]
            ), 0.19
        if self._phase in (VisualServoPhase.LOWER, VisualServoPhase.RELEASE):
            return np.array(
                [self.env.target_pos[0], self.env.target_pos[1], rest_z + 0.016]
            ), 0.19 if self._phase is VisualServoPhase.LOWER else 1.0
        return np.array(
            [self.env.target_pos[0], self.env.target_pos[1], rest_z + 0.035]
        ), 1.0

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
            feature = self.detector.detect(frame)
            if feature is None:
                self.metrics = VisualServoMetrics(failure_reason="feature_not_found")
                return Action(
                    joint_position=self._q_cmd, gripper=GripperCommand(position=1.0)
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

        if self._phase is VisualServoPhase.DESCEND:
            self._refine_from_final_camera()
        target, grip_goal = self._waypoint()
        self._grip = float(
            np.clip(grip_goal, self._grip - 0.9 * self._dt, self._grip + 0.9 * self._dt)
        )
        self._q_cmd = self._limiter(self._solve(target))
        pinch = self.env.kin.pinch_center(self.env.data)
        reached = float(np.linalg.norm(pinch - target)) <= self.ee_tolerance
        grip_now = self.env.joint_to_gripper(observation.joint_state.position[-1])
        settled_gripper = abs(grip_now - self._grip) < 0.06
        self._phase_steps += 1
        if self._phase in (VisualServoPhase.CLOSE, VisualServoPhase.RELEASE):
            self._settle_steps = self._settle_steps + 1 if settled_gripper else 0
            if self._settle_steps >= 8:
                self._advance()
        elif reached or self._phase_steps >= 120:
            self._advance()
        self.metrics = VisualServoMetrics(
            visual_error_px=None,
            ee_error_m=float(np.linalg.norm(pinch - target)),
            settled=reached,
        )
        return Action(
            joint_position=self._q_cmd, gripper=GripperCommand(position=self._grip)
        )


__all__ = [
    "CameraCalibration",
    "ColorBlobDetector",
    "SO101VisualServoPolicy",
    "VisualFeature",
    "VisualServoMetrics",
]
