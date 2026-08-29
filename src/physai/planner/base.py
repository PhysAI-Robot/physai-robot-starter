"""High-level task planner interface: language + pixels -> sub-goals.

This is the VLM box in the architecture diagram. It runs at ~0.1-1 Hz, not at
the control rate: it decomposes "cari kaleng merah di atas meja" into a short
list of grounded sub-goals, and the VLA policy executes each one closed-loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..contracts import Observation, Pose, PoseStamped, Vector3


@dataclass
class SubGoal:
    """One step of a plan.

    `skill` names the low-level behaviour the VLA policy should run; `waypoint`
    is the grounded 3-D goal in the robot base frame (the PoseStamped that
    would go to Nav2 / MoveIt in Phase 1). `target_description` keeps the
    language grounding around for the VLA's text conditioning.
    """

    skill: str
    target_description: str = ""
    waypoint: PoseStamped | None = None
    gripper: float | None = None
    rationale: str = ""

    @classmethod
    def from_xyz(cls, skill: str, xyz, **kw) -> "SubGoal":
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
    def plan(self, instruction: str, observation: Observation) -> Plan:
        ...


class ScriptedPlanner(Planner):
    """Offline stand-in for the VLM: emits the canonical pick-and-place plan.

    Use it to develop and test the planner->policy plumbing without spending
    API calls, then swap in ClaudePlanner with the same interface.
    """

    name = "scripted_planner"

    def __init__(self, pick_xyz, place_xyz) -> None:
        self.pick_xyz = np.asarray(pick_xyz, dtype=np.float64)
        self.place_xyz = np.asarray(place_xyz, dtype=np.float64)

    def plan(self, instruction: str, observation: Observation) -> Plan:
        return Plan(
            instruction=instruction,
            notes="scripted stand-in; no perception involved",
            subgoals=[
                SubGoal.from_xyz("move_above", self.pick_xyz + np.array([0, 0, 0.045]),
                                 target_description="cube", gripper=0.55),
                SubGoal.from_xyz("grasp", self.pick_xyz,
                                 target_description="cube", gripper=0.2),
                SubGoal.from_xyz("move_above", self.place_xyz + np.array([0, 0, 0.055]),
                                 target_description="target pad", gripper=0.2),
                SubGoal.from_xyz("release", self.place_xyz,
                                 target_description="target pad", gripper=0.55),
            ],
        )


class SortingPlanner(Planner):
    """Offline stand-in for a VLM that actually reads the instruction.

    Where ScriptedPlanner ignores `instruction` entirely (there is only ever
    one cube), this planner parses the color word out of it and grounds that
    to a real cube position — `env.cube_positions` stands in for what a real
    VLM would recover from the camera image. This is the layer of the
    architecture where language grounding belongs: the downstream VLA policy
    executes each sub-goal's waypoint closed-loop and never needs to see the
    instruction text itself.
    """

    name = "sorting_planner"

    def __init__(self, env, place_xyz, colors: tuple[str, ...] = ("red", "blue", "yellow")) -> None:
        self.env = env          # privileged access to cube_positions, same pattern as ScriptedPickPlace
        self.place_xyz = np.asarray(place_xyz, dtype=np.float64)
        self.colors = colors

    def _parse_color(self, instruction: str) -> str:
        lowered = instruction.lower()
        for color in self.colors:
            if color in lowered:
                return color
        raise ValueError(f"no known color ({', '.join(self.colors)}) found in instruction {instruction!r}")

    def plan(self, instruction: str, observation: Observation) -> Plan:
        color = self._parse_color(instruction)
        pick_xyz = np.asarray(self.env.cube_positions[color])
        return Plan(
            instruction=instruction,
            notes=f"parsed target color={color!r} from instruction; grounded via env.cube_positions",
            subgoals=[
                SubGoal.from_xyz("move_above", pick_xyz + np.array([0, 0, 0.045]),
                                 target_description=f"{color} cube", gripper=0.55),
                SubGoal.from_xyz("grasp", pick_xyz,
                                 target_description=f"{color} cube", gripper=0.2),
                SubGoal.from_xyz("move_above", self.place_xyz + np.array([0, 0, 0.055]),
                                 target_description="target pad", gripper=0.2),
                SubGoal.from_xyz("release", self.place_xyz,
                                 target_description="target pad", gripper=0.55),
            ],
        )
