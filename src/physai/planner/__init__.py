from .base import Plan, Planner, ScriptedPlanner, SubGoal

__all__ = ["Plan", "Planner", "ScriptedPlanner", "SubGoal", "ClaudePlanner"]


def __getattr__(name):  # lazy: don't import anthropic unless asked for
    if name == "ClaudePlanner":
        from .claude_vlm import ClaudePlanner

        return ClaudePlanner
    raise AttributeError(name)
