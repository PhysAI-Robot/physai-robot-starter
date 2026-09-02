"""Task wrapper that adds task semantics to a robot port."""

from __future__ import annotations

from typing import Any

from ..contracts import Action, Observation
from ..robots.base import RobotPort, RobotSpec
from .base import Task


class TaskRuntime:
    """Compose a task with a robot without making the robot task-aware."""

    def __init__(
        self,
        robot: RobotPort,
        task: Task,
        *,
        success_hold_steps: int = 10,
    ) -> None:
        if success_hold_steps < 1:
            raise ValueError("success_hold_steps must be positive")
        self.robot = robot
        self.task = task
        self.success_hold_steps = success_hold_steps
        self._success_streak = 0

    @property
    def robot_spec(self) -> RobotSpec:
        return self.robot.robot_spec

    def reset(self, seed: int | None = None) -> Observation:
        observation = self.robot.reset(seed=seed)
        self.task.reset(self.robot, getattr(self.robot, "rng", None))
        self._success_streak = 0
        return observation

    def observe(self) -> Observation:
        return self.robot.observe()

    def send_action(self, action: Action) -> None:
        self.robot.send_action(action)

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict]:
        observation, _, robot_terminated, truncated, robot_info = self.robot.step(action)
        info = dict(robot_info)
        info.update(self.task.evaluate(self.robot))
        self._success_streak = self._success_streak + 1 if info.get("at_target") else 0
        info["success"] = self._success_streak >= self.success_hold_steps
        terminated = robot_terminated or self.task.terminated(self.robot, info)
        return observation, self.task.reward(self.robot, info), terminated, truncated, info

    def close(self) -> None:
        self.robot.close()

    def __getattr__(self, name: str) -> Any:
        """Expose robot-specific observation helpers to Phase 0 policies."""
        return getattr(self.robot, name)


__all__ = ["TaskRuntime"]