"""Execute a Plan by driving each SubGoal through IK.

This is the glue between the VLM box and the control box in the diagram: it
turns a list of PoseStamped waypoints into joint commands. It is deliberately
*not* a VLA — it is the open-loop baseline you compare a VLA against.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np

from ..contracts import Action, GripperCommand, Observation, Pose, PoseStamped, Vector3
from ..control.resolver import JointRateLimiter
from ..planner.base import Plan, approach_from_orientation, gripper_quat_wxyz
from ..robots.base import KinematicsPort
from .base import Policy


class PlanRunner(Policy):
    name = "plan_runner"

    def __init__(
        self,
        kin: KinematicsPort,
        plan: Plan,
        *,
        pos_tol: float = 0.015,
        joint_tol: float = 0.02,
        grip_still_tol: float = 0.002,
        settle_steps: int = 10,
        max_subgoal_steps: int = 150,
        max_joint_rate: float = 0.8,
        descend_rate: float = 0.4,
        dt: float = 0.04,
        approach_dir=None,
        ik_kwargs: dict | None = None,
        locate: Callable[[str], np.ndarray | None] | None = None,
    ) -> None:
        """`locate(name)` returns where `name` ("cube") is now, or None if unknown.

        With it, the runner looks the cube up again as each approach to it begins, so a
        cube nudged since the plan was made (by the gripper rising from home, say) is
        still grasped where it is.
        """
        self.kin = kin
        # The runner moves waypoints; the caller's plan stays as it was made.
        self.plan = replace(plan, subgoals=list(plan.subgoals))
        self._locate = locate
        self.pos_tol = pos_tol
        self.joint_tol = joint_tol
        self._lag = 0.0
        self.grip_still_tol = grip_still_tol
        self._last_grip: float | None = None
        self.settle_steps = settle_steps
        self.max_subgoal_steps = max_subgoal_steps
        self._limiter = JointRateLimiter(max_joint_rate, dt)
        self._rates = (max_joint_rate * dt, descend_rate * dt)
        self.index = 0
        self._steps = 0
        self._settle = 0
        self._q_cmd: np.ndarray | None = None
        self._grip = 1.0
        self.approach_dir = approach_dir
        self.ik_kwargs = ik_kwargs or {}

    def reset(
        self,
        observation: Observation,
        goal: PoseStamped | None = None,
        instruction: str | None = None,
    ) -> None:
        self.index = 0
        self._steps = 0
        self._settle = 0
        joint_count = len(
            getattr(self.kin, "joint_names", observation.joint_state.name)
        )
        self._q_cmd = observation.joint_state.position[:joint_count].copy()
        self._grip = 1.0
        self._last_grip = None
        self._limiter.reset(self._q_cmd)

    @property
    def done(self) -> bool:
        return self.index >= len(self.plan.subgoals)

    @property
    def current(self):
        return None if self.done else self.plan.subgoals[self.index]

    def act(self, observation: Observation) -> Action:
        if self.done:
            return Action(
                joint_position=self._q_cmd, gripper=GripperCommand(position=self._grip)
            )

        if self._steps == 0:
            self._retarget(self.index)
        sg = self.plan.subgoals[self.index]
        if sg.gripper is not None:
            self._grip = float(sg.gripper)
        target = sg.waypoint.pose.position.as_array()

        # Turning the wrist top-down from the home pose is itself a sideways sweep of the
        # jaw, so a `rise` leaves the wrist as it is.
        direction, ik_kwargs = self.approach_dir, self.ik_kwargs
        asked = approach_from_orientation(sg.waypoint.pose.orientation, target[:2])
        if asked is not None:  # the plan says which way the gripper points here
            direction = asked
            quat = gripper_quat_wxyz(sg.waypoint.pose.orientation, target[:2])
            ik_kwargs = {**ik_kwargs, "target_quat_wxyz": quat}
        if sg.skill == "rise":
            direction = None
        # Slower down onto the cube: it keeps the pinch point on a straight line, so a jaw
        # does not clip the cube's edge on the way.
        self._limiter.max_delta = self._rates[sg.skill == "descend"]
        res = self.kin.ik_pinch(target, direction, q_init=self._q_cmd, **ik_kwargs)
        self._q_cmd = self._limiter(res.qpos)
        self._lag = float(np.abs(res.qpos - self._q_cmd).max())

        self._steps += 1
        return Action(
            joint_position=self._q_cmd, gripper=GripperCommand(position=self._grip)
        )

    def _retarget(self, index: int) -> None:
        """Move the cube's remaining waypoints over the cube's current x and y."""
        sg = self.plan.subgoals[index]
        if (
            self._locate is None
            or sg.skill not in ("move_above", "descend")
            or not sg.target_description
        ):
            return
        found = self._locate(sg.target_description)
        if found is None:
            return
        for j in range(index, len(self.plan.subgoals)):
            later = self.plan.subgoals[j]
            if later.target_description != sg.target_description:
                continue
            z = later.waypoint.pose.position.as_array()[2]
            xyz = np.array([found[0], found[1], z])
            pose = replace(later.waypoint.pose, position=Vector3.from_array(xyz))
            self.plan.subgoals[j] = replace(
                later, waypoint=replace(later.waypoint, pose=pose)
            )

    def note_progress(self, pinch_center: np.ndarray, gripper_now: float) -> None:
        """Advance the sub-goal cursor. Call once per control tick after `act`.

        Kept separate from `act` so the caller supplies the measured pinch
        centre (sim) or the TF lookup (ROS2) rather than this class
        reaching into the simulator.
        """
        if self.done:
            return
        sg = self.plan.subgoals[self.index]
        target = sg.waypoint.pose.position.as_array()
        # The command must also have caught up with the IK solution: the pinch point can
        # pass the waypoint while the rate-limited wrist is still turning, and descending
        # then lands a jaw on the cube.
        reached = (
            float(np.linalg.norm(pinch_center - target)) < self.pos_tol
            and self._lag < self.joint_tol
        )
        # A gripper closed on the cube stops short of its command, so "settled" also means
        # "no longer moving".
        still = (
            self._last_grip is not None
            and abs(gripper_now - self._last_grip) < self.grip_still_tol
        )
        self._last_grip = gripper_now
        grip_ok = sg.gripper is None or abs(gripper_now - sg.gripper) < 0.06 or still

        if sg.skill in ("grasp", "release"):
            # Closing or opening in mid-air is not a grasp or a release: the pinch
            # point must be at the waypoint too.
            self._settle = self._settle + 1 if grip_ok and reached else 0
            advance = self._settle >= self.settle_steps
        else:
            advance = reached and grip_ok

        if advance or self._steps >= self.max_subgoal_steps:
            self.index += 1
            self._steps = 0
            self._settle = 0
