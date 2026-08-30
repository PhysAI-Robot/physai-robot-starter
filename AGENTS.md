# Agent Guide

This file defines how coding agents should work in this repository. It is an
operating guide, not a description of the runtime architecture.

## Document ownership

Keep each document focused on one audience:

- `README.md` is the user-facing setup and workflow guide.
- `docs/ARCHITECTURE.md` is the internal design reference and the source of
  truth for module boundaries and contracts.
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
- Prefer the smallest compatible change and avoid unrelated refactors.
- Do not commit downloaded assets, model snapshots, demonstrations, videos, or
  generated plans.
- Never add credentials, API keys, or real secrets to source, tests, or docs.

## Validation

After the first substantive edit, run the narrowest relevant check immediately.
For Python changes, the default full check is:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q
```

Use a focused test path first when one exists, then run the full suite for
changes that cross module boundaries. For documentation-only changes, check
links and command names against the current files and scripts. A documentation
change must not claim a workflow that has not been verified in the repository.

## Commit messages

When creating commits, use the format `[type] imperative summary` with one
purpose per commit. Use the smallest accurate type:

- `[feat]` for a new capability or modular boundary.
- `[fix]` for a behavior correction.
- `[docs]` for documentation-only changes.
- `[test]` for test-only changes.
- `[chore]` for maintenance that does not change behavior.

Keep the summary concise and do not mix a roadmap or unrelated cleanup into a
feature commit.

## Adding a component

- New robot: add its adapter and registration, then cover its capability
  contract and generic simulation path.
- New task: keep task state, reward, metrics, and termination independent from
  robot internals.
- New planner: implement the planner contract and return the existing plan
  shape where possible.
- New policy: implement the control-rate policy contract and make its required
  observation/action capabilities explicit.

Update the architecture reference only when the supported design or ownership
has changed. Update the README only when a user-visible setup or workflow has
changed.

## Completion checklist

- The change is in the smallest owning module.
- A focused executable check has passed.
- Relevant tests or docs have been updated.
- Generated files and secrets are absent from the diff.
- Cross-document links use the exact case of the target path.