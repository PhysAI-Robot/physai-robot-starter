"""How an arm grasps a cube, and whether it can: shared by every arm's `grasp_profile` and `grasp_reach`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np


@dataclass(frozen=True)
class GraspProfile:
    """How one arm grasps a cube, as offsets (m) from the points a planner is given.

    A planner knows the cube centre and the place point; this says where the pinch point
    goes relative to them. It is the one source of those numbers for `grasp_reach`, the
    scripted planner and the plan runner.
    """

    grasp_dz: float  # pinch point above the cube centre at the grasp
    hover: float  # clearance above the grasp before descending to it
    carry: float  # lift above the grasp before the move (0 = none)
    place_hover_dz: float  # pinch height above the place point before lowering
    release_dz: float  # pinch height above the place point at the release
    grip_open: float  # gripper command, 0 closed .. 1 open
    grip_hold: float  # gripper command that holds the cube
    approach_dir: tuple[float, float, float] | None  # None leaves the wrist free
    open_descent: bool  # reach the grasp with the gripper open, then close it
    ik_kwargs: dict[str, Any] = field(default_factory=dict)  # for `kin.ik_pinch`
    # Stacking only: the arm puts a cube down with the gripper `place_tilt` degrees below
    # horizontal, pointing outward (0 = flat), after carrying it at `transit_tilt`. Set it
    # when the top-down grasp does not reach as high as the tower; a cube turned by 0 or
    # 90 degrees about the horizontal still sits square. None: place it as it was grasped.
    place_tilt: float | None = None
    transit_tilt: float = 45.0
    place_tilt_min_z: float = 0.0  # the lowest pinch height it reaches at `place_tilt`
    # Rolled by this much about the approach axis at `place_tilt` (0 at top-down, in between
    # in proportion): the jaws then close sideways, so a put-down cube is not left sitting on
    # the lower jaw, which would drag it off the tower when the gripper backs away.
    place_roll: float = 90.0

    def reach_defaults(self, rest_z: float) -> dict[str, float]:
        """The `z`, `hover` and `carry` that `grasp_reach` tests unless told otherwise."""
        return {"z": rest_z + self.grasp_dz, "hover": self.hover, "carry": self.carry}


class GraspReach(NamedTuple):
    """Which poses of one grasp the arm's IK reaches: the grasp, the hover before it, the carry after."""

    grasp: bool
    hover: bool
    carry: bool

    @property
    def ok(self) -> bool:
        return self.grasp and self.hover and self.carry


def grasp_reach_at(
    kin,
    defaults: dict[str, float],
    x: float,
    y: float,
    direction: np.ndarray,
    *,
    z: float | None,
    hover: float | None,
    carry: float | None,
    q_init: np.ndarray | None,
    **ik_kwargs,
) -> GraspReach:
    """Test an object at (x, y): the grasp, `hover` back along `direction`, `carry` above it.

    Unset `z`, `hover` and `carry` come from the robot's `defaults`; `carry` 0 skips that test.
    `q_init` is a joint-state position vector (its arm joints start the IK); `ik_kwargs` go to
    `kin.ik_pinch`.
    """
    z = defaults["z"] if z is None else z
    hover = defaults["hover"] if hover is None else hover
    carry = defaults["carry"] if carry is None else carry
    if q_init is not None:
        q_init = q_init[: len(kin.joint_names)]

    def reaches(point: np.ndarray) -> bool:
        return kin.ik_pinch(point, direction, q_init=q_init, **ik_kwargs).converged

    point = np.array([x, y, z])
    if not reaches(point):
        return GraspReach(False, False, False)
    return GraspReach(
        True,
        reaches(point - hover * direction),
        carry <= 0 or reaches(point + np.array([0.0, 0.0, carry])),
    )
