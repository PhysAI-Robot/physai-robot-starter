"""The Phase 1 ROS2 interface, declared in Phase 0.

Nothing here imports rclpy. It is the single place where topic names, message
types, and rates are written down, so that:

* Phase 0 code can be checked against the contract it will have to meet.
* Phase 1 is a mechanical port: each `Endpoint` becomes one publisher or
  subscriber, and the callback body is the Phase 0 function you already tested.

Recommended ROS2 distro: **Jazzy Jalisco** (LTS, Ubuntu 24.04, supported to
2029). Kilted Kaiju (May 2025) is non-LTS with a short support window, and
Lyrical Luth needs Ubuntu 26.04. For a multi-year research project, take the
LTS.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Direction(Enum):
    PUBLISH = "publish"
    SUBSCRIBE = "subscribe"


@dataclass(frozen=True)
class Endpoint:
    topic: str
    msg_type: str
    direction: Direction
    rate_hz: float
    owner: str
    note: str = ""


#: Node that wraps MuJoCo (Phase 0) or the real SO-101 follower (Phase 2).
ROBOT_ENDPOINTS: tuple[Endpoint, ...] = (
    Endpoint("/joint_states", "sensor_msgs/msg/JointState",
             Direction.PUBLISH, 25.0, "so101_driver",
             "5 arm joints + gripper, radians"),
    Endpoint("/camera/front/image_raw", "sensor_msgs/msg/Image",
             Direction.PUBLISH, 25.0, "so101_driver", "rgb8"),
    Endpoint("/camera/wrist/image_raw", "sensor_msgs/msg/Image",
             Direction.PUBLISH, 25.0, "so101_driver", "rgb8"),
    Endpoint("/camera/front/camera_info", "sensor_msgs/msg/CameraInfo",
             Direction.PUBLISH, 25.0, "so101_driver"),
    Endpoint("/camera/wrist/camera_info", "sensor_msgs/msg/CameraInfo",
             Direction.PUBLISH, 25.0, "so101_driver"),
    Endpoint("/tf", "tf2_msgs/msg/TFMessage",
             Direction.PUBLISH, 25.0, "so101_driver"),
    Endpoint("/arm_controller/joint_trajectory",
             "trajectory_msgs/msg/JointTrajectory",
             Direction.SUBSCRIBE, 25.0, "so101_driver",
             "absolute joint targets; what the VLA emits"),
    Endpoint("/gripper_controller/gripper_cmd",
             "control_msgs/msg/GripperCommand",
             Direction.SUBSCRIBE, 25.0, "so101_driver",
             "normalised aperture 0..1"),
)

#: Node that runs the VLA policy at the control rate.
POLICY_ENDPOINTS: tuple[Endpoint, ...] = (
    Endpoint("/joint_states", "sensor_msgs/msg/JointState",
             Direction.SUBSCRIBE, 100.0, "vla_policy"),
    Endpoint("/camera/front/image_raw", "sensor_msgs/msg/Image",
             Direction.SUBSCRIBE, 30.0, "vla_policy"),
    Endpoint("/camera/wrist/image_raw", "sensor_msgs/msg/Image",
             Direction.SUBSCRIBE, 30.0, "vla_policy"),
    Endpoint("/task/subgoal", "geometry_msgs/msg/PoseStamped",
             Direction.SUBSCRIBE, 1.0, "vla_policy",
             "current waypoint from the planner"),
    Endpoint("/task/instruction", "std_msgs/msg/String",
             Direction.SUBSCRIBE, 0.1, "vla_policy",
             "language conditioning for the VLA"),
    Endpoint("/arm_controller/joint_trajectory",
             "trajectory_msgs/msg/JointTrajectory",
             Direction.PUBLISH, 25.0, "vla_policy"),
    Endpoint("/gripper_controller/gripper_cmd",
             "control_msgs/msg/GripperCommand",
             Direction.PUBLISH, 25.0, "vla_policy"),
    Endpoint("/cmd_vel_ee", "geometry_msgs/msg/Twist",
             Direction.PUBLISH, 25.0, "vla_policy",
             "alternative Cartesian path; needs a servo node downstream"),
)

#: Topics fed from outside the robot stack (an operator UI, a bag, a test
#: script). Nothing in this graph publishes them, and that is correct.
EXTERNAL_INPUTS: frozenset[str] = frozenset({"/task/instruction"})

#: Node that runs the VLM planner. Slow, and allowed to be.
PLANNER_ENDPOINTS: tuple[Endpoint, ...] = (
    Endpoint("/camera/front/image_raw", "sensor_msgs/msg/Image",
             Direction.SUBSCRIBE, 1.0, "vlm_planner", "sampled, not streamed"),
    Endpoint("/task/instruction", "std_msgs/msg/String",
             Direction.SUBSCRIBE, 0.1, "vlm_planner"),
    Endpoint("/task/subgoal", "geometry_msgs/msg/PoseStamped",
             Direction.PUBLISH, 0.2, "vlm_planner"),
    Endpoint("/task/plan", "std_msgs/msg/String",
             Direction.PUBLISH, 0.2, "vlm_planner", "JSON plan, for logging/debug"),
)

ALL_ENDPOINTS = ROBOT_ENDPOINTS + POLICY_ENDPOINTS + PLANNER_ENDPOINTS

#: Frames, in the order they should appear in the TF tree.
TF_FRAMES: tuple[str, ...] = (
    "world", "base", "shoulder", "upper_arm", "lower_arm", "wrist",
    "gripper", "gripper_frame", "camera_front", "camera_wrist",
)

RECOMMENDED_DISTRO = "jazzy"


def describe() -> str:
    lines = [f"ROS2 contract (target distro: {RECOMMENDED_DISTRO})", ""]
    for owner in ("so101_driver", "vla_policy", "vlm_planner"):
        lines.append(f"[{owner}]")
        for e in ALL_ENDPOINTS:
            if e.owner != owner:
                continue
            arrow = "->" if e.direction is Direction.PUBLISH else "<-"
            lines.append(f"  {arrow} {e.topic:42s} {e.msg_type:44s} {e.rate_hz:6.1f} Hz")
            if e.note:
                lines.append(f"       {e.note}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
