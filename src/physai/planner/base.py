"""High-level task planner interface: language + pixels -> sub-goals.

This is the VLM box in the architecture diagram. It runs at ~0.1-1 Hz, not at
the control rate: it decomposes "cari kaleng merah di atas meja" into a short
list of grounded sub-goals, and the VLA policy executes each one closed-loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..contracts import Observation, Pose, PoseStamped, Quaternion, Vector3

if TYPE_CHECKING:
    from ..robots.reach import GraspProfile


@dataclass
class SubGoal:
    """One step of a plan.

    `skill` names the low-level behaviour the VLA policy should run; `waypoint`
    is the grounded 3-D goal in the robot base frame (the PoseStamped that
    would go to Nav2 / MoveIt on a ROS2 robot). `target_description` keeps the
    language grounding around for the VLA's text conditioning.
    """

    skill: str
    target_description: str = ""
    waypoint: PoseStamped | None = None
    gripper: float | None = None
    rationale: str = ""

    @classmethod
    def from_xyz(cls, skill: str, xyz, **kw) -> SubGoal:
        return cls(
            skill=skill,
            waypoint=PoseStamped(pose=Pose(position=Vector3.from_array(xyz))),
            **kw,
        )

    def to_dict(self) -> dict:
        return {
            "skill": self.skill,
            "target_description": self.target_description,
            "waypoint": self.waypoint.to_dict() if self.waypoint else None,
            "gripper": self.gripper,
            "rationale": self.rationale,
        }


@dataclass
class Plan:
    instruction: str
    subgoals: list[SubGoal] = field(default_factory=list)
    notes: str = ""
    raw: dict | None = None

    def __len__(self) -> int:
        return len(self.subgoals)

    def to_dict(self) -> dict:
        return {
            "instruction": self.instruction,
            "notes": self.notes,
            "subgoals": [s.to_dict() for s in self.subgoals],
        }


class Planner(ABC):
    name = "planner"

    @abstractmethod
    def plan(self, instruction: str, observation: Observation) -> Plan: ...


# skill, pinch point, target, gripper and, optionally, the tilt and roll of the gripper in degrees
# (a number or a (tilt, roll) pair, see `approach_orientation`)
Step = tuple

_TOP_DOWN = np.array([0.0, 0.0, -1.0])


def _at(base, dz: float) -> np.ndarray:
    return np.asarray(base, dtype=np.float64) + np.array([0.0, 0.0, dz])


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _spin(axis: int, angle: float) -> np.ndarray:
    """The quaternion (w, x, y, z) turning `angle` radians about world axis 0, 1 or 2."""
    q = np.zeros(4)
    q[0] = np.cos(angle / 2)
    q[1 + axis] = np.sin(angle / 2)
    return q


_TOP_DOWN_QUAT = _spin(
    1, np.pi / 2
)  # the gripper's approach axis (x) turned to point down


def approach_orientation(tilt: float, roll: float = 0.0) -> Quaternion:
    """The orientation of a gripper `tilt` degrees below horizontal, rolled by `roll` degrees.

    The orientation is given in the radial frame at the waypoint (x outward, z up) and
    relative to the top-down grasp, so the identity is that grasp and a plan that sets no
    orientation is unchanged. The frame is radial because an arm with no yaw of its own (the
    SO-101) can only point along its own radial plane: the runner turns it into a world
    orientation at the waypoint's x and y. `roll` turns the jaws about the approach axis; the
    cube they hold turns with them, and a cube turned a quarter turn still sits square.
    """
    rel = _qmul(_spin(1, np.radians(tilt)), _spin(0, np.radians(roll)))
    q = _qmul(rel, _TOP_DOWN_QUAT * np.array([1.0, -1.0, -1.0, -1.0]))
    return Quaternion(float(q[1]), float(q[2]), float(q[3]), float(q[0]))


def gripper_quat_wxyz(orientation: Quaternion, xy) -> np.ndarray | None:
    """The world orientation (w, x, y, z) a waypoint at `xy` asks of the gripper; None for the
    identity (the grasp's own)."""
    if abs(orientation.x) + abs(orientation.y) + abs(orientation.z) < 1e-9:
        return None
    rel = np.array([orientation.w, orientation.x, orientation.y, orientation.z])
    phi = float(np.arctan2(xy[1], xy[0]))
    return _qmul(_spin(2, phi), _qmul(rel, _TOP_DOWN_QUAT))


def approach_from_orientation(orientation: Quaternion, xy) -> np.ndarray | None:
    """The world approach direction a waypoint at `xy` asks for; None for the identity."""
    quat = gripper_quat_wxyz(orientation, xy)
    if quat is None:
        return None
    return Quaternion(quat[1], quat[2], quat[3], quat[0]).to_matrix()[:, 0]


def rise_steps(observation: Observation, pick_xyz, profile: GraspProfile) -> list[Step]:
    """Straight up from where the gripper is, before any sideways move.

    A direct move from the home pose sweeps the jaw through a cube near the base.
    """
    if observation.ee_pose is None:
        return []
    here = observation.ee_pose.pose.position.as_array()
    top = _at(pick_xyz, profile.grasp_dz + profile.hover)[2]
    return [("rise", np.array([here[0], here[1], top]), "", profile.grip_open)]


def pick_steps(
    pick_xyz,
    profile: GraspProfile,
    *,
    pick_name: str = "cube",
    transit_z: float | None = None,
) -> list[Step]:
    """Hover over the cube, descend, grasp, lift: the pick, at one arm's heights.

    It descends with the gripper open when the arm needs that and lifts the cube before
    carrying it. `transit_z` is the pinch height the cube is carried at when something
    stands in the way (a tower); it raises the lift to at least that.
    """
    grasp = _at(pick_xyz, profile.grasp_dz)
    steps: list[Step] = [
        ("move_above", _at(grasp, profile.hover), pick_name, profile.grip_open)
    ]
    if profile.open_descent:
        steps.append(("descend", grasp, pick_name, profile.grip_open))
    steps.append(("grasp", grasp, pick_name, profile.grip_hold))
    if profile.carry > 0 or transit_z is not None:
        lift = _at(grasp, profile.carry)
        if transit_z is not None:
            lift[2] = max(lift[2], transit_z)
        steps.append(("lift", lift, pick_name, profile.grip_hold))
    return steps


def place_steps(
    place_xyz,
    profile: GraspProfile,
    *,
    place_name: str = "target pad",
    hover_z: float | None = None,
) -> list[Step]:
    """Hover over `place_xyz`, lower the cube still held, then let go.

    Opening on the way down would drop the cube. `hover_z` replaces the profile's hover.
    """
    release = _at(place_xyz, profile.release_dz)
    hover = _at(place_xyz, profile.place_hover_dz)
    if hover_z is not None:
        hover[2] = hover_z
    return [
        ("move_above", hover, place_name, profile.grip_hold),
        ("move_above", release, place_name, profile.grip_hold),
        ("release", release, place_name, profile.grip_open),
    ]


def pick_place_steps(
    pick_xyz,
    place_xyz,
    profile: GraspProfile,
    *,
    pick_name: str = "cube",
    place_name: str = "target pad",
    transit_z: float | None = None,
    place_hover_z: float | None = None,
) -> list[Step]:
    """Pick the cube at `pick_xyz` and put it down at `place_xyz`, at one arm's heights."""
    return pick_steps(
        pick_xyz, profile, pick_name=pick_name, transit_z=transit_z
    ) + place_steps(place_xyz, profile, place_name=place_name, hover_z=place_hover_z)


def plan_from_steps(instruction: str, steps: list[Step]) -> Plan:
    subgoals = []
    for skill, xyz, what, grip, *pose in steps:
        goal = SubGoal.from_xyz(skill, xyz, target_description=what, gripper=grip)
        if pose and pose[0] is not None:
            tilt, *roll = np.atleast_1d(pose[0])
            goal.waypoint.pose.orientation = approach_orientation(tilt, *roll)
        subgoals.append(goal)
    return Plan(
        instruction=instruction,
        notes="scripted stand-in; no perception involved",
        subgoals=subgoals,
    )


class ScriptedPlanner(Planner):
    """Emits the canonical pick-and-place plan from known object poses.

    Use it to develop and test the planner->policy plumbing. A model-backed
    planner implements the same `Planner` interface.
    """

    name = "scripted_planner"

    # The gripper is position-controlled, so the squeeze force comes from how far
    # past first contact the target sits: 0.2 (barely closed on the cube) leaves
    # it slipping and the cube is pushed, 0.06 holds it through the transfer.
    GRIP_OPEN = 0.55
    GRIP_HOLD = 0.06

    def __init__(
        self, pick_xyz, place_xyz, profile: GraspProfile | None = None
    ) -> None:
        self.pick_xyz = np.asarray(pick_xyz, dtype=np.float64)
        self.place_xyz = np.asarray(place_xyz, dtype=np.float64)
        self.profile = profile

    def plan(self, instruction: str, observation: Observation) -> Plan:
        if self.profile is not None:
            return self._plan_with(instruction, self.profile, observation)
        return Plan(
            instruction=instruction,
            notes="scripted stand-in; no perception involved",
            subgoals=[
                SubGoal.from_xyz(
                    "move_above",
                    self.pick_xyz + np.array([0, 0, 0.045]),
                    target_description="cube",
                    gripper=self.GRIP_OPEN,
                ),
                SubGoal.from_xyz(
                    "grasp",
                    self.pick_xyz,
                    target_description="cube",
                    gripper=self.GRIP_HOLD,
                ),
                SubGoal.from_xyz(
                    "move_above",
                    self.place_xyz + np.array([0, 0, 0.055]),
                    target_description="target pad",
                    gripper=self.GRIP_HOLD,
                ),
                SubGoal.from_xyz(
                    "release",
                    self.place_xyz,
                    target_description="target pad",
                    gripper=self.GRIP_OPEN,
                ),
            ],
        )

    def _plan_with(
        self, instruction: str, profile: GraspProfile, observation: Observation
    ) -> Plan:
        """The same plan at the heights one arm's grasp needs (see `pick_place_steps`)."""
        steps = rise_steps(observation, self.pick_xyz, profile)
        steps += pick_place_steps(self.pick_xyz, self.place_xyz, profile)
        return plan_from_steps(instruction, steps)


# SortingPlanner (a task-coupled VLM stand-in that parses instruction text)
# is research content, kept in the private `physai-research` repo. It is
# not defined here because core must not own task-specific language
# grounding — only the Planner contract and the generic ScriptedPlanner
# baseline above do.
