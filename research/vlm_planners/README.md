# VLM planners

Model- or heuristic-grounded `Planner` implementations beyond the generic
scripted baseline (`physai.planner.base.ScriptedPlanner`, which stays in
core).

- `sorting_planner.py` — `SortingPlanner`, a task-coupled stand-in for a
  VLM-grounded planner: it parses an instruction string and reads privileged
  environment state (cube colors/positions) to produce sorting sub-goals.

Registers itself with `physai.planner.registry` on import; core never
imports this package.
