# Research

Approach-specific implementations on top of the frozen core in `src/physai/`. Each topic
owns its README (setup and runbook) and, where it has measurement history, a
`FINDINGS.md`.

## The one rule

- A `research/<topic>/` package may import `physai`'s public contracts
  (`physai.contracts`, `physai.robots.base`, `physai.policy.base`, `physai.planner.base`,
  `physai.tasks.base`, `physai.data`, the Gymnasium adapter) and the registries it
  registers itself with (`physai.policy.registry`, `physai.planner.registry`,
  `physai.robots.registry`).
- `physai` (core) never imports `research/`. Research plugs in by registering itself when
  its module is imported (enforced by import-linter,
  [DECISIONS.md A](../docs/DECISIONS.md#a-research-code-lives-outside-the-core-package)).

## Topics

| Topic | Contains |
| --- | --- |
| [`scripted_experts/`](scripted_experts/README.md) | Privileged-ground-truth experts that generate demonstrations; the project's results table |
| [`classical_control/`](classical_control/README.md) | Visual servo and other model-free closed-loop baselines |
| [`imitation_learning/`](imitation_learning/README.md) | ACT/LeRobot dataset tooling, training and checkpoint-backed policies |
| [`vlm_planners/`](vlm_planners/README.md) | Model- or heuristic-grounded `Planner` implementations beyond the scripted baseline |

Reinforcement learning and VLA research are planned (see [ROADMAP.md](../ROADMAP.md))
and get their directory when their first module exists. See
[docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md#research-boundary) for the dependency rule
and how a module registers itself.
