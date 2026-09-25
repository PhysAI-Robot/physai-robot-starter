# 3. Move research code outside the core package

## Status

Accepted

## Context

`src/physai` mixed frozen infrastructure (contracts, ports, registries) with
approach-specific research code (a scripted expert, visual servo, ACT
training, a VLA/LeRobot policy adapter, a VLM-planner stand-in) simply
because nothing ever needed a boundary between them. This made it unclear
what the core promises to keep stable versus what is one implementation
choice among many that research will keep changing.

## Decision

Create a top-level `research/<topic>/` tree, sibling to `src/`, with topics
`scripted_experts`, `classical_control`, `imitation_learning`,
`reinforcement_learning`, `vla`, `vlm_planners`. Move `robots/so101/expert.py`,
`robots/so101/visual_servo.py`, `policy/act_dataset.py`, the checkpoint-backed
half of `policy/vla_adapter.py`, `scripts/train_act.py`, and
`planner.base.SortingPlanner` there. Each research module registers itself
with a core registry (`policy.registry`, `planner.registry`,
`robots.registry`) on import. Core never imports `research/`; this is
enforced by an import-linter contract (see ADR 8).

## Consequences

Adding a new research technique never requires a core change: a new
`research/<topic>/` package plus a registration call is enough. Baselines
that are genuinely minimal and technique-agnostic (`ScriptedPlanner`,
`PlanRunner`, `SortingTask`, `ReplayPolicy`, the Gymnasium adapter, TurtleBot4's
navigation baseline) stay in core because the brief treats "minimal
baselines" as something core ships, not something research owns.
