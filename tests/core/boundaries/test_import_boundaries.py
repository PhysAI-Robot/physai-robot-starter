"""Enforce the dependency-direction rules in pyproject.toml's
[tool.importlinter] section: sim stays out of the ROS2 boundary, policy/
tasks/planner depend on ports (not MuJoCo). "Core never imports research" is
checked by an AST scan instead: `research/` is a private submodule that a plain
checkout may not have, and import-linter cannot check a package that is missing.

Calls import-linter's own `lint_imports()` (the function backing the
`lint-imports` CLI) in-process, so a plain `pytest tests/ -q` fails on a
boundary violation the same way `uv run lint-imports` would.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

importlinter_cli = pytest.importorskip("importlinter.cli")

REPO_ROOT = Path(__file__).resolve().parents[3]


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


def test_core_never_imports_research():
    offenders = []
    for path in (REPO_ROOT / "src" / "physai").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            else:
                continue
            if any(name.split(".")[0] == "research" for name in names):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert not offenders, f"src/physai must not import research: {offenders}"
