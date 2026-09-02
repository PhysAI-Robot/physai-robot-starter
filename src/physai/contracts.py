"""ROS2-shaped message contracts.

Phase 0 runs without ROS2, but every interface boundary in this repo speaks
these dataclasses instead of raw numpy. In Phase 1 each one is replaced by its
real message type with no changes to callers:

    JointState      -> sensor_msgs/msg/JointState
    Twist           -> geometry_msgs/msg/Twist
    PoseStamped     -> geometry_msgs/msg/PoseStamped
    GripperCommand  -> control_msgs/action/GripperCommand (goal)
    ImageFrame      -> sensor_msgs/msg/Image  (+ CameraInfo)

Field names and units deliberately match the ROS2 definitions (SI, radians,
quaternion as x,y,z,w).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

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
    """control_msgs/action/GripperCommand goal.

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
