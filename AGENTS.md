# Agent Guide

This file defines how coding agents should work in this repository. It is an
operating guide, not a description of the runtime architecture.

## Document ownership

Keep each document focused on one audience:

- All repository documentation must be written in English.

- `README.md` is the user-facing setup and workflow guide.
- `docs/ARCHITECTURE.md` is the internal design reference and the source of
  truth for module boundaries and contracts.
- `docs/MIGRATION.md` is the historical record of the core-architecture-freeze
  restructuring (old path -> new path); it is not a plan to execute.
- `docs/adr/` records the decisions behind the frozen design, one file per
  decision.
- `research/<topic>/README.md` is that research topic's own runbook (setup,
  commands, workflow). Detailed research workflows belong there, not in a
  `docs/*_RUNBOOK.md` file — a robot runbook links to the relevant
  `research/<topic>/README.md` instead of embedding its commands.
- `CONTRIBUTING.md` is the contributor-facing source of truth for workflow and
  commit message conventions.
- `AGENTS.md` is the agent-facing workflow, validation, and repository hygiene
  guide.
- `ROADMAP.md` records planned work and should not be treated as current
  behavior.

Do not copy detailed architecture, setup commands, or agent instructions into
the other documents. Link to the owning document instead.

## Before changing code

1. Identify the smallest owning module, symbol, or failing test.
2. Read the nearby implementation and its tests before editing.
3. Check the working tree and preserve changes that are already present.
4. State a local hypothesis about the behavior and choose the cheapest check
   that could disconfirm it.

Follow the ownership and dependency rules in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Do not move logic across those
boundaries merely to make a local test pass.

## Implementation rules

- Keep CLI files in `scripts/` thin; compose registered components there.
- Keep robot-specific behavior with the robot adapter or its backend.
- Keep task rules with the task implementation.
- Keep planner decisions behind `Planner` and low-level control behind
  `Policy`.
- Reuse the shared contracts before introducing a new message shape.
- Add an explicit adapter when a model or embodiment needs a different space;
  do not silently reshape shared arrays.
- Keep approach-specific implementations (scripted experts, visual servo,
  ACT/LeRobot, VLA checkpoints, VLM planners) under `research/<topic>/`, not
  under `src/physai`. Code in `src/physai` must never import from `research/`
  — a research module registers itself with the relevant core registry
  (`physai.robots.registry`, `physai.policy.registry`,
  `physai.planner.registry`) on import instead. This is checked by
  `uv run lint-imports` and by `tests/boundaries/test_import_boundaries.py`.
- Prefer the smallest compatible change and avoid unrelated refactors.
- Do not commit downloaded assets, model snapshots, demonstrations, videos, or
  generated plans.
- Never add credentials, API keys, or real secrets to source, tests, or docs.

## Design principles

- Apply KISS: choose the simplest design that satisfies the existing contract;
  do not add abstractions, indirection, or configuration without a concrete
  need.
- Follow SRP: give each module, class, and function one clear responsibility;
  keep behavior in the owning component described by the architecture map.
- Apply DRY: reuse shared contracts and helpers when behavior is genuinely
  shared, but do not force unrelated concepts into one abstraction merely to
  remove a small duplication.

## Validation

After the first substantive edit, run the narrowest relevant check immediately.
For Python changes, the default full check is:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/ -q
```

Use a focused test path first when one exists, then run the full suite for
changes that cross module boundaries. `tests/boundaries/test_import_boundaries.py`
runs the dependency-direction contracts in `pyproject.toml`'s
`[tool.importlinter]` section as part of that same suite; a new cross-module
import can fail there even when its own tests pass. For documentation-only
changes, check links and command names against the current files and scripts.
A documentation change must not claim a workflow that has not been verified
in the repository.

Before finishing a Python change, also run:

```bash
uv run ruff format --exclude .venv --exclude venv
```

`.github/workflows/ci.yml`'s `format` job runs `ruff format --check` on every
push and fails the build on any unformatted file; running the non-`--check`
form locally fixes formatting instead of just reporting it.

## Commit messages

Follow [CONTRIBUTING.md](CONTRIBUTING.md) for the shared commit message format
and contribution workflow. Agents use the same convention as human
contributors; do not create a second agent-only variant.

## Adding a component

Every seam below is a new file plus one registration call — see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)'s extension seam table for the
exact registry function per seam.

- New robot: build one `RobotDescriptor` and call `register_embodiment()`
  once, then cover its capability contract and generic simulation path.
- New scene: register a `SceneDefinition` declaring the robot kinds and task
  names it supports.
- New task: keep task state, reward, metrics, and termination independent
  from robot internals; register with `tasks.registry`.
- New planner: implement the `Planner` contract and return the existing plan
  shape where possible; register with `planner.registry` (a research module
  registers itself on import instead of core registering it).
- New policy: implement the control-rate policy contract and make its
  required observation/action capabilities explicit; register with
  `policy.registry`, or `robots.registry.register_robot_policy()` if it is
  owned by one robot.
- New backend: implement the adapter shape in `robots/adapters.py` and call
  `register_adapter()`.

Update the architecture reference only when the supported design or ownership
has changed. Update the README only when a user-visible setup or workflow has
changed.

## Completion checklist

- The change is in the smallest owning module.
- A focused executable check has passed.
- Relevant tests or docs have been updated.
- Generated files and secrets are absent from the diff.
- Cross-document links use the exact case of the target path.