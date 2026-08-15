"""Low-level control: turn Cartesian commands into joint targets.

Two resolvers, matching the two arrows in the architecture diagram:

* ``WaypointResolver``  — PoseStamped (from the VLM planner) -> joint targets,
  via IK. This is the "Nav2-style" open-loop goal path.
* ``TwistToJointResolver`` — Twist (/cmd_vel, from a VLA policy) -> joint
  targets, via the damped-pseudoinverse Jacobian. This is the closed-loop path.

Both emit ``Action`` so the env, and later the ROS2 joint controller, take the
same type regardless of which layer produced the command.
"""

from __future__ import annotations

import numpy as np

from ..contracts import Action, GripperCommand, JointState, PoseStamped, Twist
from ..sim.kinematics import TOP_DOWN, ArmKinematics


class JointRateLimiter:
    """Clamp joint-space jumps so IK discontinuities don't become step inputs.

    A position-actuated arm will happily be commanded to teleport; the servo
    then saturates and the cube gets knocked away. Limiting per-tick delta is
    the cheapest fix and mirrors what a real joint_trajectory_controller does.
    """

    def __init__(self, max_rate: float = 2.0, dt: float = 0.04) -> None:
        self.max_delta = max_rate * dt
        self._last: np.ndarray | None = None

    def reset(self, q: np.ndarray | None = None) -> None:
        self._last = None if q is None else np.asarray(q, dtype=np.float64).copy()

    def __call__(self, q_target: np.ndarray) -> np.ndarray:
        q_target = np.asarray(q_target, dtype=np.float64)
        if self._last is None:
            self._last = q_target.copy()
            return q_target
        delta = np.clip(q_target - self._last, -self.max_delta, self.max_delta)
        self._last = self._last + delta
        return self._last.copy()


class WaypointResolver:
    """PoseStamped -> Action (joint positions), via approach-constrained IK."""

    def __init__(
        self,
        kin: ArmKinematics,
        rate_limiter: JointRateLimiter | None = None,
        approach_dir=TOP_DOWN,
    ) -> None:
        self.kin = kin
        self.limiter = rate_limiter
        self.approach_dir = approach_dir

    def __call__(
        self,
        waypoint: PoseStamped,
        joint_state: JointState,
        gripper: GripperCommand | None = None,
    ) -> tuple[Action, float]:
        """Returns (action, position_error_metres)."""
        q_now = joint_state.position[:5]
        res = self.kin.ik(
            waypoint.pose.position.as_array(),
            self.approach_dir,
            q_init=q_now,
        )
        q = self.limiter(res.qpos) if self.limiter else res.qpos
        return Action(joint_position=q, gripper=gripper or GripperCommand()), res.position_error


class TwistToJointResolver:
    """Twist (m/s, rad/s in base frame) -> Action (joint positions).

    Integrates the commanded end-effector velocity for one control tick using a
    damped pseudo-inverse of the site Jacobian. `data` must be the live MjData
    (Phase 1: replace with a KDL/pinocchio chain fed from /joint_states).
    """

    def __init__(
        self,
        kin: ArmKinematics,
        data,
        dt: float = 0.04,
        damping: float = 0.08,
        max_joint_step: float = 0.08,
    ) -> None:
        self.kin = kin
        self.data = data
        self.dt = dt
        self.damping = damping
        self.max_joint_step = max_joint_step

    def __call__(
        self,
        twist: Twist,
        joint_state: JointState,
        gripper: GripperCommand | None = None,
    ) -> Action:
        J = self.kin.site_jacobian(self.data)          # (6, 5)
        v = twist.as_array()                           # (6,)
        JJt = J @ J.T + (self.damping ** 2) * np.eye(6)
        dq = J.T @ np.linalg.solve(JJt, v) * self.dt
        dq = np.clip(dq, -self.max_joint_step, self.max_joint_step)
        q = self.kin.clip_to_limits(joint_state.position[:5] + dq)
        return Action(joint_position=q, gripper=gripper or GripperCommand())
