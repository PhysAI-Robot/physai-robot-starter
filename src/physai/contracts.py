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

# Canonical joint ordering. Matches the SO-101 MJCF and the LeRobot follower.
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


@dataclass
class JointState:
    """sensor_msgs/msg/JointState. Positions in rad, velocities in rad/s."""

    name: tuple[str, ...] = ALL_JOINT_NAMES
    position: np.ndarray = field(default_factory=lambda: np.zeros(6))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(6))
    effort: np.ndarray = field(default_factory=lambda: np.zeros(6))
    header: Header = field(default_factory=Header)

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)
        self.effort = np.asarray(self.effort, dtype=np.float64)

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


@dataclass
class Observation:
    """Everything a policy sees at one control tick."""

    joint_state: JointState
    images: dict[str, ImageFrame] = field(default_factory=dict)
    ee_pose: PoseStamped | None = None
    step: int = 0
    sim_time: float = 0.0


@dataclass
class Action:
    """Everything a policy emits at one control tick.

    Exactly one of `joint_position` / `ee_twist` should be set. Joint position
    is the VLA-native action space (LeRobot records absolute joint targets);
    ee_twist exists for the Twist-based closed-loop path in the diagram.
    """

    joint_position: np.ndarray | None = None  # (5,) rad, arm only
    ee_twist: Twist | None = None
    gripper: GripperCommand = field(default_factory=GripperCommand)

    def __post_init__(self) -> None:
        if self.joint_position is not None:
            self.joint_position = np.asarray(self.joint_position, dtype=np.float64).reshape(-1)
