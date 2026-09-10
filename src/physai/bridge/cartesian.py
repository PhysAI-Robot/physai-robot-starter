"""Cartesian target service/action contract for fixed-base robot adapters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts import Action, GripperCommand, JointState, PoseStamped
from ..robots.base import KinematicsPort, RobotSpec


@dataclass(frozen=True)
class CartesianTargetRequest:
    """One Cartesian waypoint request, suitable for a ROS2 service/action."""

    target: PoseStamped
    gripper: GripperCommand | None = None


@dataclass(frozen=True)
class CartesianTargetResult:
    """Structured result for service responses and action feedback."""

    accepted: bool
    reason: str
    position_error: float | None = None
    orientation_error: float | None = None
    iterations: int | None = None
    action: Action | None = None


class CartesianTargetService:
    """Resolve Cartesian requests into safe joint-position actions."""

    def __init__(
        self,
        kin: KinematicsPort,
        robot_spec: RobotSpec,
        *,
        target_frame: str = "base",
    ) -> None:
        self.kin = kin
        self.robot_spec = robot_spec
        self.target_frame = target_frame

    def handle(
        self,
        request: CartesianTargetRequest,
        joint_state: JointState,
    ) -> CartesianTargetResult:
        target = request.target
        if target.header.frame_id != self.target_frame:
            return CartesianTargetResult(
                accepted=False,
                reason=f"target frame must be {self.target_frame!r}",
            )
        try:
            result = self.kin.ik(
                target.pose.position.as_array(),
                q_init=joint_state.position[:len(self.robot_spec.action_joint_names)],
                target_quat_wxyz=target.pose.orientation.to_mujoco(),
            )
        except (KeyError, ValueError, np.linalg.LinAlgError) as exc:
            return CartesianTargetResult(accepted=False, reason=str(exc))
        if not result.converged:
            return CartesianTargetResult(
                accepted=False,
                reason="IK did not converge",
                position_error=result.position_error,
                orientation_error=result.orientation_error,
                iterations=result.iterations,
            )
        action = Action(
            joint_position=result.qpos,
            joint_names=self.robot_spec.action_joint_names,
            gripper=request.gripper or GripperCommand(),
        )
        try:
            self.robot_spec.validate_action(action)
        except ValueError as exc:
            return CartesianTargetResult(
                accepted=False,
                reason=str(exc),
                position_error=result.position_error,
                orientation_error=result.orientation_error,
                iterations=result.iterations,
            )
        return CartesianTargetResult(
            accepted=True,
            reason="accepted",
            position_error=result.position_error,
            orientation_error=result.orientation_error,
            iterations=result.iterations,
            action=action,
        )


__all__ = [
    "CartesianTargetRequest",
    "CartesianTargetResult",
    "CartesianTargetService",
]
