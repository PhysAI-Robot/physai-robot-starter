"""Privileged SO-101 pick-and-place expert for demonstration collection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from ...contracts import Action, GripperCommand, Observation, PoseStamped
from ...control.resolver import JointRateLimiter
from ...policy.base import Policy
from ..base import KinematicsPort


class Phase(Enum):
    APPROACH = auto()
    DESCEND = auto()
    CLOSE = auto()
    SQUEEZE = auto()
    LIFT = auto()
    TRANSFER = auto()
    LOWER = auto()
    RELEASE = auto()
    RETREAT = auto()
    DONE = auto()


@dataclass
class ExpertConfig:
    approach_dir: np.ndarray | None = None
    hover_height: float = 0.045
    grasp_height: float = 0.000
    lift_height: float = 0.035
    place_height: float = 0.016
    gripper_open: float = 0.55
    gripper_touch: float = 0.21
    gripper_grip: float = 0.19
    pos_tol: float = 0.012
    settle_steps: int = 8
    max_phase_steps: int = 120
    max_joint_rate: float = 1.2
    approach_rate: float = 0.5
    gripper_rate: float = 0.9


class SO101PickPlaceExpert(Policy):
    """Privileged SO-101 policy used to bootstrap visual-control datasets."""

    name = "scripted_pick_place"

    def __init__(
        self, kin: KinematicsPort, env, cfg: ExpertConfig | None = None
    ) -> None:
        self.kin = kin
        self.env = env
        self.cfg = cfg or ExpertConfig()
        self.phase = Phase.APPROACH
        self._phase_steps = 0
        self._settle = 0
        self._q_cmd: np.ndarray | None = None
        self._grip = self.cfg.gripper_open
        self._grasp_xy: np.ndarray | None = None
        control_hz = float(env.robot_spec.metadata.get("control_hz", 25.0))
        self._dt = 1.0 / control_hz
        self._limiter = JointRateLimiter(self.cfg.max_joint_rate, self._dt)
        self._joint_count = len(
            getattr(kin, "joint_names", env.robot_spec.action_joint_names)
        )
        self._gripper_index = env.robot_spec.joint_names.index("gripper")

    def reset(
        self,
        observation: Observation,
        goal: PoseStamped | None = None,
        instruction: str | None = None,
    ) -> None:
        self.phase = Phase.APPROACH
        self._phase_steps = 0
        self._settle = 0
        self._q_cmd = observation.joint_state.position[: self._joint_count].copy()
        self._grip = self.cfg.gripper_open
        self._grasp_xy = None
        self._limiter = JointRateLimiter(self.cfg.max_joint_rate, self._dt)
        self._limiter.reset(self._q_cmd)

    @property
    def done(self) -> bool:
        return self.phase is Phase.DONE

    def _solve(self, target_xyz: np.ndarray) -> np.ndarray:
        res = self.kin.ik_pinch(target_xyz, self.cfg.approach_dir, q_init=self._q_cmd)
        if not res.converged:
            return self._q_cmd
        return res.qpos

    def _waypoint(self) -> tuple[np.ndarray, float]:
        cfg = self.cfg
        target = self.env.target_pos
        table_top = self.env.cfg.scene.table_pos[2] + self.env.cfg.scene.table_size[2]
        rest_z = table_top + self.env.cfg.scene.cube_half

        if self.phase in (Phase.APPROACH, Phase.DESCEND, Phase.CLOSE, Phase.SQUEEZE):
            cube = self.env.cube_pos
            if self._grasp_xy is None:
                self._grasp_xy = cube[:2].copy()
            gx, gy = self._grasp_xy
            if self.phase is Phase.APPROACH:
                return np.array([gx, gy, cube[2] + cfg.hover_height]), cfg.gripper_open
            grip = {
                Phase.DESCEND: cfg.gripper_open,
                Phase.CLOSE: cfg.gripper_touch,
                Phase.SQUEEZE: cfg.gripper_grip,
            }[self.phase]
            return np.array([gx, gy, cube[2] + cfg.grasp_height]), grip

        carry_z = rest_z + cfg.lift_height
        if self.phase is Phase.LIFT:
            gx, gy = (
                self._grasp_xy if self._grasp_xy is not None else self.env.cube_pos[:2]
            )
            return np.array([gx, gy, carry_z]), cfg.gripper_grip
        if self.phase is Phase.TRANSFER:
            return np.array([target[0], target[1], carry_z]), cfg.gripper_grip
        if self.phase is Phase.LOWER:
            return np.array(
                [target[0], target[1], rest_z + cfg.place_height]
            ), cfg.gripper_grip
        if self.phase is Phase.RELEASE:
            return np.array(
                [target[0], target[1], rest_z + cfg.place_height]
            ), cfg.gripper_open
        return np.array([target[0], target[1], carry_z]), cfg.gripper_open

    def _advance(self) -> None:
        order = [
            Phase.APPROACH,
            Phase.DESCEND,
            Phase.CLOSE,
            Phase.SQUEEZE,
            Phase.LIFT,
            Phase.TRANSFER,
            Phase.LOWER,
            Phase.RELEASE,
            Phase.RETREAT,
            Phase.DONE,
        ]
        self.phase = order[min(order.index(self.phase) + 1, len(order) - 1)]
        self._phase_steps = 0
        self._settle = 0

    def act(self, observation: Observation) -> Action:
        if self.phase is Phase.DONE:
            return Action(
                joint_position=self._q_cmd,
                gripper=GripperCommand(position=self._grip),
            )

        target_xyz, grip_goal = self._waypoint()
        step = self.cfg.gripper_rate * self._dt
        self._grip = float(np.clip(grip_goal, self._grip - step, self._grip + step))
        rate = (
            self.cfg.max_joint_rate
            if self.phase is Phase.APPROACH
            else self.cfg.approach_rate
        )
        self._limiter.max_delta = rate * self._dt
        self._q_cmd = self._limiter(self._solve(target_xyz))

        pinch = self.kin.pinch_center(self.env.data)
        reached = float(np.linalg.norm(pinch - target_xyz)) < self.cfg.pos_tol
        grip_now = self.env.joint_to_gripper(
            observation.joint_state.position[self._gripper_index]
        )
        grip_vel = abs(float(observation.joint_state.velocity[self._gripper_index]))
        ramp_done = abs(self._grip - grip_goal) < 1e-6
        gripper_settled = ramp_done and (
            abs(grip_now - self._grip) < 0.06 or grip_vel < 0.05
        )

        self._phase_steps += 1
        if self.phase in (Phase.CLOSE, Phase.SQUEEZE, Phase.RELEASE):
            if gripper_settled:
                self._settle += 1
            if self._settle >= self.cfg.settle_steps:
                self._advance()
        elif reached or self._phase_steps >= self.cfg.max_phase_steps:
            self._advance()

        return Action(
            joint_position=self._q_cmd,
            gripper=GripperCommand(position=self._grip),
        )


__all__ = ["ExpertConfig", "Phase", "SO101PickPlaceExpert"]
