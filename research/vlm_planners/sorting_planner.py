"""Offline stand-in for a VLM that actually reads the instruction.

Research module: this is task-coupled (it depends on a sorting task's
privileged ``env.cube_positions``) and is not a generic baseline, so it
lives outside core. Core never imports this module directly (see
research/README.md).
"""

from __future__ import annotations

import numpy as np

from physai.contracts import Observation
from physai.planner.base import Plan, Planner, SubGoal
from physai.planner.registry import register_planner


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

    def __init__(
        self, env, place_xyz, colors: tuple[str, ...] = ("red", "blue", "yellow")
    ) -> None:
        self.env = env  # privileged access to cube_positions for scripted baselines
        self.place_xyz = np.asarray(place_xyz, dtype=np.float64)
        self.colors = colors

    def _parse_color(self, instruction: str) -> str:
        lowered = instruction.lower()
        for color in self.colors:
            if color in lowered:
                return color
        raise ValueError(
            f"no known color ({', '.join(self.colors)}) found in instruction {instruction!r}"
        )

    def plan(self, instruction: str, observation: Observation) -> Plan:
        color = self._parse_color(instruction)
        pick_xyz = np.asarray(self.env.cube_positions[color])
        return Plan(
            instruction=instruction,
            notes=f"parsed target color={color!r} from instruction; grounded via env.cube_positions",
            subgoals=[
                SubGoal.from_xyz(
                    "move_above",
                    pick_xyz + np.array([0, 0, 0.045]),
                    target_description=f"{color} cube",
                    gripper=0.55,
                ),
                SubGoal.from_xyz(
                    "grasp", pick_xyz, target_description=f"{color} cube", gripper=0.2
                ),
                SubGoal.from_xyz(
                    "move_above",
                    self.place_xyz + np.array([0, 0, 0.055]),
                    target_description="target pad",
                    gripper=0.2,
                ),
                SubGoal.from_xyz(
                    "release",
                    self.place_xyz,
                    target_description="target pad",
                    gripper=0.55,
                ),
            ],
        )


register_planner("sorting_planner", SortingPlanner)


__all__ = ["SortingPlanner"]
