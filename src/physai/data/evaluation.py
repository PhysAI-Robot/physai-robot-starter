"""Shared evaluation result and aggregate reporting contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class EvaluationReport:
    policy: str
    robot: str
    task: str
    results: tuple[Mapping[str, Any], ...]

    @property
    def summary(self) -> dict[str, Any]:
        results = list(self.results)
        n = len(results)
        success = sum(bool(result.get("success")) for result in results)
        collision = sum(bool(result.get("collision")) for result in results)
        timeout = sum(bool(result.get("timeout")) for result in results)
        unsafe = sum(bool(result.get("unsafe_action")) for result in results)
        held_out = [result for result in results if result.get("held_out")]
        return {
            "episodes": n,
            "success_count": success,
            "success_rate": success / n if n else 0.0,
            "collision_count": collision,
            "timeout_count": timeout,
            "unsafe_action_count": unsafe,
            "mean_reward": float(
                np.mean([result.get("reward", 0.0) for result in results])
            )
            if n
            else 0.0,
            "mean_steps": float(np.mean([result.get("steps", 0) for result in results]))
            if n
            else 0.0,
            "held_out_episodes": len(held_out),
            "held_out_success_rate": (
                sum(bool(result.get("success")) for result in held_out) / len(held_out)
                if held_out
                else None
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "physai.evaluation.v1",
            "policy": self.policy,
            "robot": self.robot,
            "task": self.task,
            "summary": self.summary,
            "results": [dict(result) for result in self.results],
        }


__all__ = ["EvaluationReport"]
