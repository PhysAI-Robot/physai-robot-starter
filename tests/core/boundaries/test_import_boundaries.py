"""Enforce the dependency-direction rules in pyproject.toml's
[tool.importlinter] section: sim stays out of the ROS2 boundary, policy/
tasks/planner depend on ports (not MuJoCo). "Core never imports research" is
checked by an AST scan instead: `research/` is a separate repository that a plain
checkout does not have, and import-linter cannot check a package that is missing.

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
    """Neither the package, the scripts nor the core tests may import `research`:
    a plugin is loaded by name from a manifest's `plugins:`, never by core code."""
    offenders = []
    core_files = [
        *(REPO_ROOT / "src" / "physai").rglob("*.py"),
        *(REPO_ROOT / "scripts").glob("*.py"),
        *(REPO_ROOT / "tests" / "core").rglob("*.py"),
    ]
    for path in core_files:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            else:
                continue
            if any(name.split(".")[0] == "research" for name in names):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert not offenders, f"core must not import research: {offenders}"


# Scripts that may name one robot's package, and why. A script that drives a robot takes a
# manifest and reads `session.runtime.robot` (see scripts/_common_args.py); everything else
# belongs behind the robot's own methods, not in a script.
ROBOT_SPECIFIC_SCRIPTS = {
    "send_nav_goal.py": "Nav2 acceptance result of the TurtleBot",
}


def test_scripts_do_not_import_one_robots_package():
    robots = REPO_ROOT / "src" / "physai" / "robots"
    packages = {p.name for p in robots.iterdir() if (p / "__init__.py").exists()}
    offenders = []
    for path in (REPO_ROOT / "scripts").glob("*.py"):
        if path.name in ROBOT_SPECIFIC_SCRIPTS:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                module = node.module or ""
                names = [module, *(f"{module}.{alias.name}" for alias in node.names)]
            else:
                continue
            for name in names:
                parts = name.split(".")
                if (
                    parts[:2] == ["physai", "robots"]
                    and parts[2:3]
                    and parts[2] in packages
                ):
                    offenders.append(f"{path.name}:{node.lineno} imports {name}")
    assert not offenders, (
        "scripts must stay robot-agnostic (take a manifest, or add the behavior to the "
        f"robot and call it); allowed exceptions: {sorted(ROBOT_SPECIFIC_SCRIPTS)}: {offenders}"
    )
