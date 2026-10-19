"""The SO-101 demo policy: core's minimal proof that a policy drives the arm.

It plans from the scene's true cube and target positions, at the heights the arm's
`grasp_profile` gives: the canonical pick-and-place with `ScriptedPlanner`, or the
whole stack with `StackPlanner` when the task is `stack_cubes`. `PlanRunner` executes
the plan (IK per sub-goal) and looks each cube up again before it descends. It is
privileged: a smoke test of the policy seam and the default of the core scripts, not a
baseline to compare research against.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...contracts import Action, Observation, PoseStamped
from ...planner import ScriptedPlanner, StackPlanner
from ...policy.base import Policy
from ...policy.plan_runner import PlanRunner


# A cube further than this from the pinch point is not the one in the gripper, in metres.
HELD_REACH = 0.05


class SO101DemoPolicy(Policy):
    name = "demo"

    def __init__(self, env: Any) -> None:
        self.env = env
        self._runner: PlanRunner | None = None

    def _held_offset(self) -> np.ndarray:
        """Where the held cube sits relative to the pinch point, in x and y.

        The jaws do not hold a cube exactly on the pinch point, and a cube put down by the
        pinch point lands that far off: on a tower it can be enough to miss it.
        """
        pinch = self.env.pinch_center()
        held = min(
            self.env.cube_positions.values(),
            key=lambda c: float(np.linalg.norm(c - pinch)),
        )
        if np.linalg.norm(held - pinch) > HELD_REACH:
            return np.zeros(2)
        return (held - pinch)[:2]

    def _locate(self, name: str):
        """Where a named thing is now, as the pinch point must be sent to deal with it.

        "cube" is the task's cube, "red cube" the red one. "on red cube" is the spot over it
        and "pad" the pad, where a cube is put down: those are shifted by how the held cube
        sits in the jaws.
        """
        if name == "cube":
            return self.env.cube_pos
        if name == "pad":
            where = np.asarray(self.env.target_pos, dtype=np.float64).copy()
        elif name.startswith("on "):
            where = self.env.cube_positions.get(
                name.removeprefix("on ").removesuffix(" cube")
            )
        else:
            return self.env.cube_positions.get(name.removesuffix(" cube"))
        if where is None:
            return None
        where = np.asarray(where, dtype=np.float64).copy()
        where[:2] -= self._held_offset()
        return where

    def reset(
        self,
        observation: Observation,
        goal: PoseStamped | None = None,
        instruction: str | None = None,
    ) -> None:
        profile = self.env.grasp_profile()
        task = getattr(self.env, "task", None)
        if task is not None and task.name == "stack_cubes":
            planner = StackPlanner(
                self.env.cube_positions,
                self.env.target_pos,
                task.order,
                self.env.cube_half,
                self.env.table_top,
                profile,
            )
        else:
            planner = ScriptedPlanner(self.env.cube_pos, self.env.target_pos, profile)
        plan = planner.plan(instruction or "pick and place", observation)
        self._runner = PlanRunner(
            self.env.kin,
            plan,
            dt=1.0 / self.env.cfg.control_hz,
            approach_dir=profile.approach_dir,
            ik_kwargs=profile.ik_kwargs,
            locate=self._locate,
        )
        self._runner.reset(observation)

    def act(self, observation: Observation) -> Action:
        if self._runner is None:
            self.reset(observation)
        # Progress is measured on the state this observation came from, the
        # same cadence as scripts/plan_task.py (step, then note progress).
        self._runner.note_progress(
            self.env.pinch_center(),
            self.env.joint_to_gripper(
                observation.joint_state.position[len(self.env.kin.joint_names)]
            ),
        )
        return self._runner.act(observation)


def make_demo_policy(*, env: Any, **_: Any) -> SO101DemoPolicy:
    return SO101DemoPolicy(env)


__all__ = ["SO101DemoPolicy", "make_demo_policy"]
