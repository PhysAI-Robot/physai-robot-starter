from .base import Plan, Planner, ScriptedPlanner, SubGoal
from .registry import available_planners, create_planner, register_planner

__all__ = [
    "Plan", "Planner", "ScriptedPlanner", "SubGoal", "ClaudePlanner",
    "SmolVLMPlanner", "available_planners", "create_planner", "register_planner",
]


def __getattr__(name):  # lazy: don't import anthropic unless asked for
    if name == "ClaudePlanner":
        from .claude_vlm import ClaudePlanner

        return ClaudePlanner
    if name == "SmolVLMPlanner":
        from .smolvlm import SmolVLMPlanner

        return SmolVLMPlanner
    raise AttributeError(name)
