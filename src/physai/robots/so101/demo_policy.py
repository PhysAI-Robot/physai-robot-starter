"""The SO-101 demo policy: core's minimal proof that a policy drives the arm.

It plans the canonical pick-and-place with `ScriptedPlanner` from the scene's
true cube and target positions and executes the plan with `PlanRunner` (IK per
sub-goal). It is privileged and open-loop: a smoke test of the policy seam and
the default of the core scripts, not a baseline to compare research against.
"""

from __future__ import annotations

from typing import Any

from ...contracts import Action, Observation, PoseStamped
from ...planner import ScriptedPlanner
from ...policy.base import Policy
from ...policy.plan_runner import PlanRunner


class SO101DemoPolicy(Policy):
    name = "demo"

    def __init__(self, env: Any) -> None:
        self.env = env
        self._runner: PlanRunner | None = None

    def reset(
        self,
        observation: Observation,
        goal: PoseStamped | None = None,
        instruction: str | None = None,
    ) -> None:
        planner = ScriptedPlanner(self.env.cube_pos, self.env.target_pos)
        plan = planner.plan(instruction or "pick and place", observation)
        self._runner = PlanRunner(self.env.kin, plan, dt=1.0 / self.env.cfg.control_hz)
        self._runner.reset(observation)

    def act(self, observation: Observation) -> Action:
        if self._runner is None:
            self.reset(observation)
        # Progress is measured on the state this observation came from, the
        # same cadence as scripts/plan_task.py (step, then note progress).
        self._runner.note_progress(
            self.env.pinch_center(),
            self.env.joint_to_gripper(observation.joint_state.position[5]),
        )
        return self._runner.act(observation)


def make_demo_policy(*, env: Any, **_: Any) -> SO101DemoPolicy:
    return SO101DemoPolicy(env)


__all__ = ["SO101DemoPolicy", "make_demo_policy"]
