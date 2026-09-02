"""Scripted pick-and-place expert — the demonstration generator for Phase 0.

This is a privileged policy: it reads the cube pose straight out of the sim
instead of from pixels. That is exactly what you want for bootstrapping a VLA —
it produces the (image, state, action) triples that SmolVLA/ACT/TurboVLA train
on, without a human teleoperating 50 episodes first.

State machine:

    APPROACH -> DESCEND -> CLOSE -> LIFT -> TRANSFER -> LOWER -> RELEASE -> RETREAT

Kinematic note: the SO-101 has no shoulder roll, so a strict top-down approach
is only reachable within roughly 5 cm of the table. `hover_height` is set
accordingly — raise it and IK will start failing to hold the approach axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from ..contracts import Action, GripperCommand, Observation, PoseStamped
from ..control.resolver import JointRateLimiter
from ..robots.base import KinematicsPort
from ..robots.so101.kinematics import TOP_DOWN
from .base import Policy


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
    hover_height: float = 0.045     # above the cube centre, top-down reachable
    grasp_height: float = 0.000     # pinch centre level with the cube centre
    lift_height: float = 0.055
    place_height: float = 0.016
    # Closing happens in two stages. Slamming straight to `gripper_grip` sweeps
    # the free cube out of the jaws before they capture it; stopping at
    # `gripper_touch` (≈ the object width) captures it with ~zero squeeze, and
    # only then is it safe to squeeze down for real holding force.
    gripper_open: float = 0.55      # normalised aperture; > cube width
    gripper_touch: float = 0.21     # jaw gap ≈ 30 mm, just wider than the cube
    gripper_grip: float = 0.18      # a few mm of squeeze; sets holding force
    pos_tol: float = 0.012
    settle_steps: int = 8           # ticks to hold at CLOSE / RELEASE
    max_phase_steps: int = 120
    # Without rate limiting the position servos are handed a step input and the
    # arm swats the cube off the table before it ever closes the jaws.
    max_joint_rate: float = 1.2     # rad/s
    approach_rate: float = 0.5      # rad/s, used from DESCEND onwards
    # The jaws close at ~5 rad/s if commanded as a step, which bats a free cube
    # out of the gripper before contact can capture it. Normalised units/s.
    gripper_rate: float = 0.9


class ScriptedPickPlace(Policy):
    name = "scripted_pick_place"

    def __init__(self, kin: KinematicsPort, env, cfg: ExpertConfig | None = None) -> None:
        self.kin = kin
        self.env = env          # for privileged cube/target pose access
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

    # -- lifecycle -----------------------------------------------------
    def reset(self, observation: Observation, goal: PoseStamped | None = None,
              instruction: str | None = None) -> None:
        self.phase = Phase.APPROACH
        self._phase_steps = 0
        self._settle = 0
        self._q_cmd = observation.joint_state.position[:5].copy()
        self._grip = self.cfg.gripper_open
        self._grasp_xy = None
        self._limiter = JointRateLimiter(self.cfg.max_joint_rate, self._dt)
        self._limiter.reset(self._q_cmd)

    @property
    def done(self) -> bool:
        return self.phase is Phase.DONE

    # -- helpers -------------------------------------------------------
    def _solve(self, target_xyz: np.ndarray) -> np.ndarray:
        """Waypoints address the *pinch centre*, not the EE site — the object
        must end up between the jaws, not on top of the fixed one."""
        res = self.kin.ik_pinch(target_xyz, TOP_DOWN, q_init=self._q_cmd)
        if not res.converged:
            return self._q_cmd
        return res.qpos

    def _waypoint(self) -> tuple[np.ndarray, float]:
        """Return (target xyz for the *pinch centre*, gripper aperture)."""
        cfg = self.cfg
        t = self.env.target_pos
        table_top = self.env.cfg.scene.table_pos[2] + self.env.cfg.scene.table_size[2]
        rest_z = table_top + self.env.cfg.scene.cube_half   # cube centre on table

        # Pre-grasp phases track the live cube. Post-grasp phases must NOT:
        # the cube moves with the gripper, so a live-tracking target recedes as
        # fast as the arm approaches it and the phase never completes.
        if self.phase in (Phase.APPROACH, Phase.DESCEND, Phase.CLOSE, Phase.SQUEEZE):
            c = self.env.cube_pos
            if self._grasp_xy is None:
                self._grasp_xy = c[:2].copy()
            gx, gy = self._grasp_xy
            if self.phase is Phase.APPROACH:
                return np.array([gx, gy, c[2] + cfg.hover_height]), cfg.gripper_open
            grip = {
                Phase.DESCEND: cfg.gripper_open,
                Phase.CLOSE: cfg.gripper_touch,
                Phase.SQUEEZE: cfg.gripper_grip,
            }[self.phase]
            return np.array([gx, gy, c[2] + cfg.grasp_height]), grip

        carry_z = rest_z + cfg.lift_height
        if self.phase is Phase.LIFT:
            gx, gy = self._grasp_xy if self._grasp_xy is not None else self.env.cube_pos[:2]
            return np.array([gx, gy, carry_z]), cfg.gripper_grip
        if self.phase is Phase.TRANSFER:
            return np.array([t[0], t[1], carry_z]), cfg.gripper_grip
        if self.phase is Phase.LOWER:
            return np.array([t[0], t[1], rest_z + cfg.place_height]), cfg.gripper_grip
        if self.phase is Phase.RELEASE:
            return np.array([t[0], t[1], rest_z + cfg.place_height]), cfg.gripper_open
        # RETREAT / DONE
        return np.array([t[0], t[1], carry_z]), cfg.gripper_open

    def _advance(self) -> None:
        order = [
            Phase.APPROACH, Phase.DESCEND, Phase.CLOSE, Phase.SQUEEZE, Phase.LIFT,
            Phase.TRANSFER, Phase.LOWER, Phase.RELEASE, Phase.RETREAT, Phase.DONE,
        ]
        self.phase = order[min(order.index(self.phase) + 1, len(order) - 1)]
        self._phase_steps = 0
        self._settle = 0

    # -- policy --------------------------------------------------------
    def act(self, observation: Observation) -> Action:
        if self.phase is Phase.DONE:
            return Action(joint_position=self._q_cmd,
                          gripper=GripperCommand(position=self._grip))

        target_xyz, grip_goal = self._waypoint()
        step = self.cfg.gripper_rate * self._dt
        self._grip = float(np.clip(grip_goal, self._grip - step, self._grip + step))
        grip = self._grip
        # Slow down once we're near the cube — a fast descent bounces it away.
        rate = (self.cfg.max_joint_rate if self.phase is Phase.APPROACH
                else self.cfg.approach_rate)
        self._limiter.max_delta = rate * self._dt
        self._q_cmd = self._limiter(self._solve(target_xyz))

        pinch = self.kin.pinch_center(self.env.data)
        reached = float(np.linalg.norm(pinch - target_xyz)) < self.cfg.pos_tol
        # A squeeze never reaches its commanded aperture — the object blocks it.
        # Treat "jaws have stopped moving" as settled, not "jaws arrived".
        grip_now = self.env.joint_to_gripper(observation.joint_state.position[5])
        grip_vel = abs(float(observation.joint_state.velocity[5]))
        ramp_done = abs(self._grip - grip_goal) < 1e-6
        gripper_settled = ramp_done and (
            abs(grip_now - grip) < 0.06 or grip_vel < 0.05
        )

        self._phase_steps += 1
        if self.phase in (Phase.CLOSE, Phase.SQUEEZE, Phase.RELEASE):
            # Hold still while the jaws move; position tolerance is irrelevant.
            if gripper_settled:
                self._settle += 1
            if self._settle >= self.cfg.settle_steps:
                self._advance()
        elif reached:
            self._advance()

        return Action(joint_position=self._q_cmd,
                      gripper=GripperCommand(position=self._grip))
