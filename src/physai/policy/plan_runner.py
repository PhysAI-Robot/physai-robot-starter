"""Execute a Plan by driving each SubGoal through IK.

This is the glue between the VLM box and the control box in the diagram: it
turns a list of PoseStamped waypoints into joint commands. It is deliberately
*not* a VLA — it is the open-loop baseline you compare a VLA against.
"""

from __future__ import annotations

import numpy as np

from ..contracts import Action, GripperCommand, Observation, PoseStamped
from ..control.resolver import JointRateLimiter
from ..planner.base import Plan
from ..robots.base import KinematicsPort
from ..robots.so101.kinematics import TOP_DOWN
from .base import Policy


class PlanRunner(Policy):
    name = "plan_runner"

    def __init__(
        self,
        kin: KinematicsPort,
        plan: Plan,
        *,
        pos_tol: float = 0.015,
        settle_steps: int = 10,
        max_subgoal_steps: int = 150,
        max_joint_rate: float = 0.8,
        dt: float = 0.04,
    ) -> None:
        self.kin = kin
        self.plan = plan
        self.pos_tol = pos_tol
        self.settle_steps = settle_steps
        self.max_subgoal_steps = max_subgoal_steps
        self._limiter = JointRateLimiter(max_joint_rate, dt)
        self.index = 0
        self._steps = 0
        self._settle = 0
        self._q_cmd: np.ndarray | None = None
        self._grip = 1.0

    def reset(self, observation: Observation, goal: PoseStamped | None = None,
              instruction: str | None = None) -> None:
        self.index = 0
        self._steps = 0
        self._settle = 0
        self._q_cmd = observation.joint_state.position[:5].copy()
        self._grip = 1.0
        self._limiter.reset(self._q_cmd)

    @property
    def done(self) -> bool:
        return self.index >= len(self.plan.subgoals)

    @property
    def current(self):
        return None if self.done else self.plan.subgoals[self.index]

    def act(self, observation: Observation) -> Action:
        if self.done:
            return Action(joint_position=self._q_cmd,
                          gripper=GripperCommand(position=self._grip))

        sg = self.plan.subgoals[self.index]
        if sg.gripper is not None:
            self._grip = float(sg.gripper)
        target = sg.waypoint.pose.position.as_array()

        res = self.kin.ik_pinch(target, TOP_DOWN, q_init=self._q_cmd)
        self._q_cmd = self._limiter(res.qpos)

        self._steps += 1
        return Action(joint_position=self._q_cmd,
                      gripper=GripperCommand(position=self._grip))

    def note_progress(self, pinch_center: np.ndarray, gripper_now: float) -> None:
        """Advance the sub-goal cursor. Call once per control tick after `act`.

        Kept separate from `act` so the caller supplies the measured pinch
        centre (sim) or the TF lookup (Phase 1 / ROS2) rather than this class
        reaching into the simulator.
        """
        if self.done:
            return
        sg = self.plan.subgoals[self.index]
        target = sg.waypoint.pose.position.as_array()
        reached = float(np.linalg.norm(pinch_center - target)) < self.pos_tol
        grip_ok = sg.gripper is None or abs(gripper_now - sg.gripper) < 0.06

        if sg.skill in ("grasp", "release"):
            self._settle = self._settle + 1 if grip_ok else 0
            advance = self._settle >= self.settle_steps
        else:
            advance = reached and grip_ok

        if advance or self._steps >= self.max_subgoal_steps:
            self.index += 1
            self._steps = 0
            self._settle = 0
