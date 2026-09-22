# 7. Add planner and backend registries

## Status

Accepted

## Context

Robot, scene, task, and policy all had a register/create registry pair, but
planners were constructed directly by callers (no lookup-by-name), and
backend/adapter selection was a string switch inside
`robots/adapters.py`'s `select_adapter()` rather than a registry. This meant
two of the seven extension seams listed in the architecture did not actually
meet the "one new file + one registration call" bar.

## Decision

Add `planner/registry.py` (`register_planner` / `create_planner` /
`available_planners`), mirroring `tasks/registry.py` exactly. Extend
`robots/adapters.py` with `register_adapter` / `create_adapter` wrapping
today's `select_adapter()` switch, so `direct_mujoco` / `ros2_mujoco` /
`ros2_hardware` become registry entries.

## Consequences

A new planner or backend is now a registration call like every other seam.
`ScriptedPlanner` registers from core; `SortingPlanner` registers itself from
`research/vlm_planners/` on import, without core ever importing it.
