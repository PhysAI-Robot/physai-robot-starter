# Research

This tree holds every approach-specific implementation on top of the frozen
core in `src/physai/`: scripted experts, classical control, imitation
learning, reinforcement learning, VLA, and VLM planners.

## The one rule

- A `research/<topic>/` package may import `physai`'s public contracts
  (`physai.contracts`, `physai.robots.base`, `physai.policy.base`,
  `physai.planner.base`, `physai.tasks.base`, `physai.data`, the Gymnasium
  adapter) and the registries it needs to register itself
  (`physai.policy.registry`, `physai.planner.registry`, `physai.robots.registry`).
- `physai` (core) must never import anything from `research/`. Core ships
  contracts and minimal baselines only; research plugs in by registering
  itself when its own module is imported, never the other way around.

## Topics

| Topic | Contains |
| --- | --- |
| `scripted_experts/` | Privileged-ground-truth expert policies used to generate demonstrations |
| `classical_control/` | Visual servo and other classical closed-loop control baselines |
| `imitation_learning/` | ACT/LeRobot dataset tooling, training, and checkpoint-backed policies |
| `reinforcement_learning/` | Deep RL training on top of the Gymnasium adapter (no module yet) |
| `vla/` | Vision-language-action research beyond the adapter contract (no module yet) |
| `vlm_planners/` | Model- or heuristic-grounded `Planner` implementations beyond the scripted baseline |

Each topic owns its own README with setup notes and, where relevant, a
roadmap. See [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) for the
core/research dependency rule and how a research module registers itself.
