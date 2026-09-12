"""Shared application contracts and ROS2-compatible value types.

These dataclasses are the internal boundary between policies, tasks, robot
ports, and adapters. Their fields and units mirror the corresponding ROS2
messages, but they remain transport-neutral so direct MuJoCo execution does
not require ROS2. The bridge converts them at the ROS2 boundary:

    JointState      -> sensor_msgs/msg/JointState
    Twist           -> geometry_msgs/msg/Twist
    PoseStamped     -> geometry_msgs/msg/PoseStamped
    GripperCommand  -> control_msgs/msg/GripperCommand
    ImageFrame      -> sensor_msgs/msg/Image  (+ CameraInfo)

Field names and units deliberately match the ROS2 definitions (SI, radians,
quaternion as x,y,z,w).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

# SO-101 compatibility names. Other embodiments must provide their own ordering.
ARM_JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
GRIPPER_JOINT_NAME = "gripper"
ALL_JOINT_NAMES: tuple[str, ...] = ARM_JOINT_NAMES + (GRIPPER_JOINT_NAME,)


def _now() -> float:
    return time.time()


@dataclass
class Header:
    """std_msgs/msg/Header."""

    stamp: float = field(default_factory=_now)
    frame_id: str = ""

    def validate(self, *, require_frame: bool = False) -> None:
        if not np.isfinite(self.stamp):
            raise ValueError("header timestamp must be finite")
        if require_frame and not self.frame_id:
            raise ValueError("header frame_id must not be empty")


@dataclass
class JointState:
    """sensor_msgs/msg/JointState. Positions in rad, velocities in rad/s."""

    name: tuple[str, ...] = ALL_JOINT_NAMES
    position: np.ndarray = field(default_factory=lambda: np.zeros(6))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(6))
    effort: np.ndarray = field(default_factory=lambda: np.zeros(6))
    header: Header = field(default_factory=Header)

    def __post_init__(self) -> None:
        self.name = tuple(self.name)
        self.position = np.asarray(self.position, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)
        self.effort = np.asarray(self.effort, dtype=np.float64)
        sizes = {self.position.size, self.velocity.size, self.effort.size}
        if len(sizes) != 1 or self.position.size != len(self.name):
            raise ValueError("joint names and state arrays must have the same size")
        if len(set(self.name)) != len(self.name):
            raise ValueError("joint names must be unique")
        if not all(np.isfinite(values).all()
                   for values in (self.position, self.velocity, self.effort)):
            raise ValueError("joint state contains non-finite values")

    def validate(
        self,
        *,
        expected_names: tuple[str, ...] | None = None,
        expected_frame: str | None = None,
    ) -> None:
        self.header.validate(require_frame=expected_frame is not None)
        if expected_names is not None and self.name != expected_names:
            raise ValueError(
                f"joint state expects names {expected_names}, got {self.name}"
            )
        if expected_frame is not None and self.header.frame_id != expected_frame:
            raise ValueError(
                f"joint state expects frame {expected_frame!r}, "
                f"got {self.header.frame_id!r}"
            )

    def get(self, joint: str) -> float:
        return float(self.position[self.name.index(joint)])

    def to_dict(self) -> dict:
        return {
            "name": list(self.name),
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "effort": self.effort.tolist(),
            "stamp": self.header.stamp,
        }


@dataclass
class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @classmethod
    def from_array(cls, a) -> "Vector3":
        a = np.asarray(a, dtype=np.float64).reshape(3)
        return cls(float(a[0]), float(a[1]), float(a[2]))


@dataclass
class Quaternion:
    """Note the ROS field order: x, y, z, w (MuJoCo stores w first)."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.w], dtype=np.float64)

    @classmethod
    def from_mujoco(cls, wxyz) -> "Quaternion":
        w, x, y, z = (float(v) for v in wxyz)
        return cls(x, y, z, w)

    def to_mujoco(self) -> np.ndarray:
        return np.array([self.w, self.x, self.y, self.z], dtype=np.float64)


@dataclass
class Pose:
    position: Vector3 = field(default_factory=Vector3)
    orientation: Quaternion = field(default_factory=Quaternion)


@dataclass
class PoseStamped:
    """geometry_msgs/msg/PoseStamped — the VLM planner's waypoint output."""

    pose: Pose = field(default_factory=Pose)
    header: Header = field(default_factory=lambda: Header(frame_id="base"))

    def to_dict(self) -> dict:
        return {
            "frame_id": self.header.frame_id,
            "position": self.pose.position.as_array().tolist(),
            "orientation_xyzw": self.pose.orientation.as_array().tolist(),
        }


@dataclass
class Twist:
    """geometry_msgs/msg/Twist — end-effector velocity command (/cmd_vel style).

    linear in m/s, angular in rad/s, expressed in the base frame.
    """

    linear: Vector3 = field(default_factory=Vector3)
    angular: Vector3 = field(default_factory=Vector3)
    frame_id: str = "base"

    def as_array(self) -> np.ndarray:
        return np.concatenate([self.linear.as_array(), self.angular.as_array()])

    @classmethod
    def from_array(cls, a) -> "Twist":
        a = np.asarray(a, dtype=np.float64).reshape(6)
        return cls(Vector3.from_array(a[:3]), Vector3.from_array(a[3:]))


@dataclass
class GripperCommand:
    """control_msgs/msg/GripperCommand command.

    `position` is normalized aperture: 0.0 = fully closed, 1.0 = fully open.
    The env maps it onto the gripper joint's actual range.
    """

    position: float = 1.0
    max_effort: float = 0.0

    def clipped(self) -> float:
        return float(np.clip(self.position, 0.0, 1.0))


@dataclass
class ImageFrame:
    """sensor_msgs/msg/Image (rgb8) plus the bits of CameraInfo we care about."""

    data: np.ndarray  # (H, W, 3) uint8
    camera_name: str = ""
    header: Header = field(default_factory=Header)

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    encoding: str = "rgb8"

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data)
        if self.data.ndim != 3 or self.data.shape[2] != 3:
            raise ValueError("image data must have shape (height, width, 3)")
        if self.data.dtype != np.uint8:
            raise ValueError("image data must use uint8 values")
        self.data = np.ascontiguousarray(self.data)
        if not self.camera_name:
            raise ValueError("image camera_name must not be empty")

    def validate(
        self,
        *,
        expected_camera: str | None = None,
        expected_frame: str | None = None,
    ) -> None:
        self.header.validate(require_frame=True)
        if self.encoding != "rgb8":
            raise ValueError(f"unsupported image encoding {self.encoding!r}")
        if expected_camera is not None and self.camera_name != expected_camera:
            raise ValueError(
                f"image expects camera {expected_camera!r}, "
                f"got {self.camera_name!r}"
            )
        if expected_frame is not None and self.header.frame_id != expected_frame:
            raise ValueError(
                f"image {self.camera_name!r} expects frame {expected_frame!r}, "
                f"got {self.header.frame_id!r}"
            )


@dataclass
class Observation:
    """Everything a policy sees at one control tick."""

    joint_state: JointState
    images: dict[str, ImageFrame] = field(default_factory=dict)
    ee_pose: PoseStamped | None = None
    step: int = 0
    sim_time: float = 0.0

    def validate(
        self,
        *,
        expected_joint_names: tuple[str, ...] | None = None,
        expected_joint_frame: str | None = None,
        expected_camera_frames: dict[str, str] | None = None,
    ) -> None:
        self.joint_state.validate(
            expected_names=expected_joint_names,
            expected_frame=expected_joint_frame,
        )
        if not isinstance(self.step, int) or self.step < 0:
            raise ValueError("observation step must be a non-negative integer")
        if not np.isfinite(self.sim_time) or self.sim_time < 0:
            raise ValueError("observation sim_time must be finite and non-negative")
        camera_frames = expected_camera_frames or {}
        for name, frame in self.images.items():
            if not isinstance(frame, ImageFrame):
                raise ValueError(f"observation image {name!r} is not an ImageFrame")
            frame.validate(
                expected_camera=name,
                expected_frame=camera_frames.get(name),
            )
        if self.ee_pose is not None:
            self.ee_pose.header.validate(require_frame=True)


@dataclass
class Action:
    """Everything a policy emits at one control tick.

    A policy may emit either joint targets or a Cartesian/base twist. The
    selected robot capability decides which representation is valid.
    """

    joint_position: np.ndarray | None = None  # embodiment-defined joint targets
    ee_twist: Twist | None = None
    gripper: GripperCommand | None = field(default_factory=GripperCommand)
    joint_names: tuple[str, ...] | None = None
    stamp: float | None = None

    def __post_init__(self) -> None:
        if self.joint_position is not None:
            self.joint_position = np.asarray(self.joint_position, dtype=np.float64).reshape(-1)
            if self.joint_names is not None:
                self.joint_names = tuple(self.joint_names)

    @property
    def mode(self) -> str | None:
        """Return the action representation carried by this command."""
        if self.joint_position is not None and self.ee_twist is not None:
            raise ValueError("Action cannot contain both joint_position and ee_twist")
        if self.joint_position is not None:
            return "joint_position"
        if self.ee_twist is not None:
            return "twist"
        return None


@dataclass(frozen=True)
class TensorSpec:
    """Canonical metadata for one numeric observation or action value."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    units: str = "unitless"
    minimum: float | None = None
    maximum: float | None = None
    normalization: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tensor spec name must not be empty")
        if any(not isinstance(size, int) or size < 0 for size in self.shape):
            raise ValueError("tensor spec shape must contain non-negative integers")
        np.dtype(self.dtype)
        if self.minimum is not None and not np.isfinite(self.minimum):
            raise ValueError("tensor spec minimum must be finite")
        if self.maximum is not None and not np.isfinite(self.maximum):
            raise ValueError("tensor spec maximum must be finite")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("tensor spec minimum must not exceed maximum")

    def validate(self, value: Any) -> np.ndarray:
        array = np.asarray(value)
        if array.shape != self.shape:
            raise ValueError(
                f"{self.name!r} expects shape {self.shape}, got {array.shape}"
            )
        if array.dtype != np.dtype(self.dtype):
            raise ValueError(
                f"{self.name!r} expects dtype {self.dtype!r}, got {array.dtype!s}"
            )
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"{self.name!r} must use a numeric dtype")
        if not np.isfinite(array).all():
            raise ValueError(f"{self.name!r} contains non-finite values")
        if self.minimum is not None and np.any(array < self.minimum):
            raise ValueError(f"{self.name!r} contains values below its minimum")
        if self.maximum is not None and np.any(array > self.maximum):
            raise ValueError(f"{self.name!r} contains values above its maximum")
        return array

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "units": self.units,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "normalization": dict(self.normalization),
        }


@dataclass(frozen=True)
class CameraSpec:
    """Canonical metadata for one image observation stream."""

    name: str
    shape: tuple[int, int, int]
    dtype: str = "uint8"
    encoding: str = "rgb8"
    frame_id: str = ""
    normalization: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("camera spec name must not be empty")
        if len(self.shape) != 3 or any(size <= 0 for size in self.shape):
            raise ValueError("camera spec shape must be (height, width, channels)")
        np.dtype(self.dtype)
        if not self.encoding:
            raise ValueError("camera spec encoding must not be empty")

    def validate(self, value: Any) -> np.ndarray:
        array = np.asarray(value)
        if array.shape != self.shape:
            raise ValueError(
                f"camera {self.name!r} expects shape {self.shape}, got {array.shape}"
            )
        if array.dtype != np.dtype(self.dtype):
            raise ValueError(
                f"camera {self.name!r} expects dtype {self.dtype!r}, got {array.dtype!s}"
            )
        return array

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "encoding": self.encoding,
            "frame_id": self.frame_id,
            "normalization": dict(self.normalization),
        }


@dataclass(frozen=True)
class ObservationSpec:
    """Canonical training schema for policy observations."""

    fields: tuple[TensorSpec, ...] = ()
    cameras: tuple[CameraSpec, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = [spec.name for spec in (*self.fields, *self.cameras)]
        if len(names) != len(set(names)):
            raise ValueError("observation spec names must be unique")

    def validate(self, values: Mapping[str, Any]) -> None:
        expected = {spec.name for spec in (*self.fields, *self.cameras)}
        missing = expected - values.keys()
        extra = values.keys() - expected
        if missing:
            raise ValueError(f"observation is missing fields: {sorted(missing)}")
        if extra:
            raise ValueError(f"observation has unexpected fields: {sorted(extra)}")
        for spec in (*self.fields, *self.cameras):
            spec.validate(values[spec.name])

    def to_dict(self) -> dict[str, Any]:
        return {
            "fields": [spec.to_dict() for spec in self.fields],
            "cameras": [spec.to_dict() for spec in self.cameras],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ActionSpec:
    """Canonical training schema for policy actions."""

    fields: tuple[TensorSpec, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = [spec.name for spec in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("action spec names must be unique")

    def validate(self, values: Mapping[str, Any]) -> None:
        expected = {spec.name for spec in self.fields}
        missing = expected - values.keys()
        extra = values.keys() - expected
        if missing:
            raise ValueError(f"action is missing fields: {sorted(missing)}")
        if extra:
            raise ValueError(f"action has unexpected fields: {sorted(extra)}")
        for spec in self.fields:
            spec.validate(values[spec.name])

    def to_dict(self) -> dict[str, Any]:
        return {
            "fields": [spec.to_dict() for spec in self.fields],
            "metadata": dict(self.metadata),
        }
