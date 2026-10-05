# Repository scope and boundary decisions

ADRs 3, 5 and 8: what lives where, and how the rules are enforced.

## ADR 3: Research code lives outside the core package

Status: accepted.

**Context.** `src/physai` mixed frozen infrastructure (contracts, ports,
registries) with approach-specific code (scripted expert, visual servo, ACT
training, a VLA adapter, a VLM-planner stand-in), so it was unclear what the core
promised to keep stable.

**Decision.** A top-level `research/<topic>/` tree beside `src/`: topics
`scripted_experts`, `classical_control`, `imitation_learning` and
`vlm_planners` (further topics are added when their first module exists). Each
research module registers itself with a core registry (`policy.registry`,
`planner.registry`, `robots.registry`) on import. Core never imports `research/`;
ADR 8 enforces it.

**Consequences.** A new technique needs a new package and a registration call, no
core change. Minimal, technique-agnostic baselines stay in core (`ScriptedPlanner`,
`PlanRunner`, `SortingTask`, `ReplayPolicy`, the Gymnasium adapter, TurtleBot4
navigation).

## ADR 5: TurtleBot4 stays as the second embodiment

Status: accepted.

**Context.** SO-101 is the focus. TurtleBot4 (mobile base, twist actions) already
works; removing it would shrink the multi-embodiment surface, and developing it
would compete with SO-101 for attention.

**Decision.** It stays as the proof that the `RobotSpec` capability abstraction
generalizes beyond an arm, with its navigation baseline
(`robots/turtlebot/navigation.py`) as its one registered capability. It is not a
development focus and no new TurtleBot4 research is planned.

**Consequences.** Its tests and the ROS2/Nav2 acceptance path keep passing. A thin
feature set is intentional scope, not neglect.

## ADR 8: Enforce dependency direction with import-linter

Status: accepted.

**Context.** The architecture stated dependency rules (`sim` must not import ROS2,
`policy`/`tasks`/`planner` must not import MuJoCo, core must not import `research/`)
but nothing enforced them, and one violation existed.

**Decision.** `import-linter` with contracts in `pyproject.toml`'s
`[tool.importlinter]`, run by `lint-imports` and wired into
`tests/core/boundaries/test_import_boundaries.py`, so `pytest tests/ -q` fails on a
violation. It is a declarative, maintained tool rather than a hand-written
import-graph walker.

**Consequences.** The rules are machine-checked, not convention. The tool is never
imported by `physai`; it is a base dependency since
[ADR 16](simulators.md#adr-16-isaacsim-as-a-project-extra-not-a-separate-venv)
folded the dev tooling into the base install. Later ADRs add contracts (the
description and workspace modules stay simulator-agnostic, the engines stay
independent).
