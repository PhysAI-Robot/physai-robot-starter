"""Enforce the dependency-direction rules in pyproject.toml's
[tool.importlinter] section: sim stays out of the ROS2 boundary, policy/
tasks/planner depend on ports (not MuJoCo), and core never imports research.

Calls import-linter's own `lint_imports()` (the function backing the
`lint-imports` CLI) in-process, so a plain `pytest tests/ -q` fails on a
boundary violation the same way `uv run lint-imports` would.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

importlinter_cli = pytest.importorskip("importlinter.cli")

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_dependency_direction_contracts_hold():
    original_cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        exit_status = importlinter_cli.lint_imports(no_cache=True, no_logo=True)
    finally:
        os.chdir(original_cwd)
    assert exit_status == importlinter_cli.EXIT_STATUS_SUCCESS, (
        "import-linter found a dependency-direction violation; "
        "run `uv run lint-imports` for the report "
        "(see pyproject.toml's [tool.importlinter] section)"
    )
