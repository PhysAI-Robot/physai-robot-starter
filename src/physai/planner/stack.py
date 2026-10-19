"""Planner for stacking cubes: decides the order and the height of every placement."""

from __future__ import annotations

import numpy as np

from ..contracts import Observation
from .base import (
    Plan,
    Planner,
    _at,
    pick_place_steps,
    pick_steps,
    plan_from_steps,
    rise_steps,
)

# Gap kept between a carried cube's underside and the tower, and between the open jaws and
# the tower top after a release, in metres.
CARRY_CLEARANCE = 0.015
RETREAT_CLEARANCE = 0.03
# Just above the release height: the cube is carried over the tower, then lowered a little.
PLACE_HOVER = 0.01
# Where the cube hovers over the tower before it is lowered, when it is put down from the side.
TILT_HOVER = 0.02
# A cube held by a horizontal gripper sits this much above the pinch point, in metres, so the
# pinch point goes that much lower for the cube to land at its level instead of dropping.
TILT_RELEASE_DROP = 0.02
# Extra height, in metres, for the turns of the wrist, which move the held cube about.
TILT_TURN_ROOM = 0.03
# Degrees the gripper turns between two waypoints.
TURN_STEP = 15.0


class StackPlanner(Planner):
    """Pick each cube in `order` and put it on the pad, one level above the last.

    Like `ScriptedPlanner` it reads the true cube positions, so it plans without
    perception; what it adds is the decisions: the order, the height of each level, the
    height cubes are carried over the tower at, and a retreat above the tower between cubes
    so the next approach does not sweep it. The heights come from the tower and the arm's
    grasp, not from fixed hovers: an arm that grasps top-down reaches only so high, and
    every extra centimetre of hover is one it may not have.
    """

    name = "stack_planner"

    def __init__(
        self,
        cube_positions: dict[str, np.ndarray],
        pad_xyz,
        order: tuple[str, ...],
        cube_half: float,
        table_top: float,
        profile,
    ) -> None:
        self.cube_positions = cube_positions
        self.pad_xyz = np.asarray(pad_xyz, dtype=np.float64)
        self.order = tuple(order)
        self.cube_half = cube_half
        self.table_top = table_top
        self.profile = profile

    def _tower_top(self, cubes: int) -> float:
        return self.table_top + 2 * self.cube_half * cubes

    def _pose(self, tilt: float) -> tuple[float, float]:
        """The tilt and the roll that goes with it: none at top-down, `place_roll` at the flat."""
        return (tilt, self.profile.place_roll * (90.0 - tilt) / 90.0)

    def _turn(self, at, start: float, end: float, grip: float, name: str = "") -> list:
        """Turn the gripper from `start` to `end` degrees in place, `TURN_STEP` at a time.

        One jump between two poses moves the joints along a straight line in joint space, and
        the pinch point sags along the way; small turns keep it where it is.
        """
        sign = 1.0 if end > start else -1.0
        tilts = [*np.arange(start, end, sign * TURN_STEP)[1:], end]
        return [
            ("move_above", at, name, grip, self._pose(float(tilt))) for tilt in tilts
        ]

    def _tilted_level(self, level: int, colour: str, last: bool) -> list:
        """One cube, put down with the gripper turned toward the horizontal.

        The top-down grasp stops reaching at some height, so a tall tower is built from
        the side: pick top-down, turn the wrist above the cube to `transit_tilt`, carry the
        cube over, turn to `place_tilt` above the tower, lower, let go, back off up and turn
        back toward top-down for the next pick.
        """
        profile, half = self.profile, self.cube_half
        place = _at(self.pad_xyz, 2 * half * level)
        release_z = max(
            place[2] + profile.release_dz - TILT_RELEASE_DROP, profile.place_tilt_min_z
        )
        clear = (
            max(self._tower_top(level), self._tower_top(1))
            + half
            + profile.grasp_dz
            + CARRY_CLEARANCE
        )
        # Turning the wrist lets the held cube slip below the pinch point; leave room for it.
        carry_z = max(release_z + TILT_HOVER, clear) + TILT_TURN_ROOM
        pick = self.cube_positions[colour]
        lift_z = _at(pick, profile.grasp_dz)[2] + profile.hover
        flat, transit = profile.place_tilt, profile.transit_tilt
        name = "pad" if level == 0 else f"on {self.order[level - 1]} cube"
        hold, opened = profile.grip_hold, profile.grip_open
        over = lambda z: np.array([place[0], place[1], z])  # noqa: E731
        above_pick = np.array([pick[0], pick[1], carry_z])
        steps = [
            *pick_steps(pick, profile, pick_name=f"{colour} cube", transit_z=lift_z),
            *self._turn(above_pick, 90.0, transit, hold),
            ("move_above", over(carry_z), "", hold, self._pose(float(transit))),
            *self._turn(over(carry_z), transit, flat, hold),
            (
                "move_above",
                over(release_z + TILT_HOVER),
                name,
                hold,
                self._pose(float(flat)),
            ),
            ("move_above", over(release_z), name, hold, self._pose(float(flat))),
            ("release", over(release_z), name, opened, self._pose(float(flat))),
            ("move_above", over(carry_z), "", opened, self._pose(float(flat))),
        ]
        if not last:
            steps += self._turn(over(carry_z), flat, 75.0, opened)
        return steps

    def plan(self, instruction: str, observation: Observation) -> Plan:
        profile = self.profile
        steps = rise_steps(observation, self.cube_positions[self.order[0]], profile)
        if profile.place_tilt is not None:
            for level, colour in enumerate(self.order):
                last = level == len(self.order) - 1
                steps += self._tilted_level(level, colour, last)
            return plan_from_steps(instruction, steps)
        for level, colour in enumerate(self.order):
            place = _at(self.pad_xyz, 2 * self.cube_half * level)
            release_z = place[2] + profile.release_dz
            # The carried cube's underside is `grasp_dz + cube_half` below the pinch point.
            # It must clear the tower and also the cubes still lying on the table.
            clear = (
                max(self._tower_top(level), self._tower_top(1))
                + self.cube_half
                + profile.grasp_dz
                + CARRY_CLEARANCE
            )
            transit = max(release_z + PLACE_HOVER, clear)
            steps += pick_place_steps(
                self.cube_positions[colour],
                place,
                profile,
                pick_name=f"{colour} cube",
                # Level 0 goes on the pad. Each later cube goes on the one below it, wherever
                # that cube ended up: a cube set down on the pad can settle a few cm off.
                place_name="pad" if level == 0 else f"on {self.order[level - 1]} cube",
                transit_z=transit,
                place_hover_z=transit,
            )
            if level < len(self.order) - 1:
                retreat = (
                    self._tower_top(level + 1)
                    + self.cube_half
                    + profile.grasp_dz
                    + RETREAT_CLEARANCE
                )
                steps.append(
                    (
                        "move_above",
                        np.array([place[0], place[1], retreat]),
                        "",
                        profile.grip_open,
                    )
                )
        return plan_from_steps(instruction, steps)
