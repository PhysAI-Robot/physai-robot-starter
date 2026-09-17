"""Instruction-to-plan implementations and planner registry exports."""

from .base import Plan, Planner, ScriptedPlanner, SortingPlanner, SubGoal
from .registry import available_planners, create_planner, register_planner

__all__ = [
    "ClaudePlanner",
    "Plan",
    "Planner",
    "ScriptedPlanner",
    "SmolVLMPlanner",
    "SortingPlanner",
    "SubGoal",
    "available_planners",
    "create_planner",
    "register_planner",
]


def __getattr__(name):  # lazy: don't import anthropic unless asked for
    if name == "ClaudePlanner":
        from .claude_vlm import ClaudePlanner

        return ClaudePlanner
    if name == "SmolVLMPlanner":
        from .smolvlm import SmolVLMPlanner

        return SmolVLMPlanner
    raise AttributeError(name)
