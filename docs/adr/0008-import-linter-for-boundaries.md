# 8. Enforce dependency direction with import-linter

## Status

Accepted

## Context

The architecture document stated four dependency-direction rules (`sim` must
not import ROS2; `policy`/`tasks`/`planner` must not import MuJoCo or a
concrete robot; clients must not import `sim`/`robots` directly; core must
not import `research/`) but nothing enforced them. Violations were only
caught by manual review — and one existed
(`web/world_runtime.py` importing `robots/so101` internals directly).

## Decision

Add [`import-linter`](https://import-linter.readthedocs.io/) as a dev-only
dependency (`pyproject.toml`'s `dev` extra) rather than a hand-rolled
AST-walking test. It is a mature, widely used tool with a declarative
contract format, which is more reliable and less maintenance than
reimplementing import-graph analysis by hand. Contracts are declared in
`pyproject.toml`'s `[tool.importlinter]` section and run via `lint-imports`,
wired into `tests/boundaries/test_import_boundaries.py` so a plain
`pytest tests/ -q` fails on a violation.

## Consequences

This is a new **dev/CI-only** dependency, not a runtime dependency — it is
never imported by `physai` itself and is not required to run the framework,
only to develop it. The four rules above become machine-checked instead of
convention-only.
