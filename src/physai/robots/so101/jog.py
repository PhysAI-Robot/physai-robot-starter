"""SO-101 keyboard-jog resolution.

Shared by the standalone env (`env.py`, the single-robot web-viewer path)
and `SO101SharedInstance` (`shared.py`, the `--world` multi-robot path) so
the merge/clamp math that turns one jog tick into a full 5-element
`Action.joint_position` lives in exactly one place.
"""

from __future__ import annotations

import numpy as np

from ...contracts import Action, GripperCommand, JointState, Twist, Vector3
from ...control.resolver import TwistToJointResolver
from .contracts import ARM_JOINT_NAMES

SHOULDER_PAN_INDEX = ARM_JOINT_NAMES.index("shoulder_pan")
WRIST_FLEX_INDEX = ARM_JOINT_NAMES.index("wrist_flex")
WRIST_ROLL_INDEX = ARM_JOINT_NAMES.index("wrist_roll")
# shoulder_lift, elbow_flex -- the two joints the Cartesian (x/z) jog solves
# over. shoulder_pan is excluded and jogged directly instead (see below):
# mixing base rotation into the position solve was ill-conditioned enough
# (a translation target can demand sweeping the whole base through a wide
# arc) to be both sluggish and, via the resulting fast swings, the biggest
# source of dynamic coupling into the wrist joints.
CARTESIAN_INDICES = slice(1, 3)

# A direct-jog joint's target can advance every tick even while the servo
# is still catching up to it (e.g. a sustained hold ramped to top speed
# outruns what the actuator can physically track against gravity/inertia).
# Left unchecked the gap grows without bound, and `SafetyController`'s
# max-step gate (0.5-0.75 rad, see robot_spec.max_joint_delta) then rejects
# the next command outright. Capping how far the target may lead the live
# reading keeps jogging responsive while guaranteeing every command this
# resolver produces stays inside that gate.
MAX_TARGET_LEAD = 0.35


def _bounded(candidate: float, live: float, limits: tuple[float, float]) -> float:
    candidate = min(max(candidate, live - MAX_TARGET_LEAD), live + MAX_TARGET_LEAD)
    return float(np.clip(candidate, *limits))


def _advance_direct_joint(
    current: float, live: float, rate: float, dt: float, limits: tuple[float, float]
) -> float:
    return _bounded(current + rate * dt, live, limits)


def resolve_jog(
    twist: Twist,
    joint_state: JointState,
    *,
    cartesian_resolver: TwistToJointResolver,
    target: np.ndarray,
    shoulder_pan_limits: tuple[float, float],
    wrist_flex_limits: tuple[float, float],
    wrist_roll_limits: tuple[float, float],
    dt: float,
    gripper: GripperCommand | None = None,
) -> Action:
    """Resolve one jog tick into a full 5-joint `Action`.

    `target` is the caller's persistent 5-joint jog setpoint (mutated in
    place and reused across ticks), used as the integration base instead of
    the live `joint_state`: a moving joint dynamically couples a small nudge
    into every other joint each tick, and integrating from the live reading
    unconditionally re-affirms each nudge as the new target, ratcheting a
    joint that isn't being actively jogged away from where it was actually
    told to stay -- this is what previously let W/S/A/D visibly drag
    `wrist_flex` over a sustained hold. The caller resyncs `target` itself
    (e.g. on reset) when something other than this resolver has moved the
    robot; this function never reads a "how stale is it" heuristic from the
    live position, because a fast Cartesian jog can legitimately push the
    servo's own tracking error well past what a plausible staleness
    threshold would be.

    `twist.linear.x`/`.z` are solved through `cartesian_resolver`, a
    `TwistToJointResolver` built over a 2-joint kinematics instance
    (`shoulder_lift`, `elbow_flex`) targeting a site at the wrist, so the
    Cartesian jog can only move those two joints.

    `twist.angular.x`/`.y`/`.z` are repurposed as direct `shoulder_pan`/
    `wrist_flex`/`wrist_roll` joint-rate commands (rad/s) -- bypassing the
    Jacobian entirely, so those joints move only when explicitly jogged,
    never as a side effect of the Cartesian solve.
    """
    cartesian_twist = Twist(
        linear=twist.linear, angular=Vector3(), frame_id=twist.frame_id
    )
    cartesian_action = cartesian_resolver(
        cartesian_twist,
        joint_state,
        gripper=None,
        base_position=target[CARTESIAN_INDICES],
    )
    live_cartesian = joint_state.position[CARTESIAN_INDICES]
    previous_cartesian = target[CARTESIAN_INDICES].copy()
    target[CARTESIAN_INDICES] = np.clip(
        cartesian_action.joint_position,
        live_cartesian - MAX_TARGET_LEAD,
        live_cartesian + MAX_TARGET_LEAD,
    )
    # shoulder_lift and elbow_flex are both pitch joints on parallel axes, so
    # the gripper's world-frame pitch is exactly their sum -- confirmed
    # empirically (perturbing either alone, or both, produces exactly that
    # much rotation, and wrist_flex's own axis cancels it 1:1). Feeding back
    # -(this tick's shoulder_lift/elbow_flex change) into wrist_flex keeps
    # the gripper pointing the same direction (straight down, from a normal
    # start pose) through x/z jogging, like an excavator bucket's linkage
    # auto-levelling as the boom and stick move, rather than the gripper
    # tipping over as the arm reconfigures. `angular.y` (I/K) still layers a
    # manual tilt on top of this baseline.
    auto_level_correction = -float(
        target[CARTESIAN_INDICES].sum() - previous_cartesian.sum()
    )

    target[SHOULDER_PAN_INDEX] = _advance_direct_joint(
        target[SHOULDER_PAN_INDEX],
        joint_state.position[SHOULDER_PAN_INDEX],
        twist.angular.x,
        dt,
        shoulder_pan_limits,
    )
    target[WRIST_FLEX_INDEX] = _bounded(
        target[WRIST_FLEX_INDEX] + twist.angular.y * dt + auto_level_correction,
        joint_state.position[WRIST_FLEX_INDEX],
        wrist_flex_limits,
    )
    target[WRIST_ROLL_INDEX] = _advance_direct_joint(
        target[WRIST_ROLL_INDEX],
        joint_state.position[WRIST_ROLL_INDEX],
        twist.angular.z,
        dt,
        wrist_roll_limits,
    )
    return Action(joint_position=target.copy(), gripper=gripper or GripperCommand())


__all__ = ["resolve_jog"]
