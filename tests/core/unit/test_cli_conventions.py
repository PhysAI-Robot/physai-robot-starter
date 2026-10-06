"""The scripts' flags follow the table in docs/ARCHITECTURE.md ("CLI conventions").

Scans `scripts/*.py` with `ast` (no script is imported, so no simulator is needed).
"""

import ast
from pathlib import Path

import pytest

SCRIPTS = sorted((Path(__file__).resolve().parents[3] / "scripts").glob("*.py"))

# Old name -> the one name that replaced it.
RENAMED = {
    "--out-dir": "--out",
    "--dest": "--out",
    "--save-plan": "--out",
    "--save-frames": "--out",
    "--json-out": "--json",
    "--merged-out": "--json",
    "--dataset-dir": "--dataset",
    "--record-dir": "--dataset",
    "--video-dir": "--out",
    "--video-name": "--name",
    "--max-ticks": "--max-steps",
}


def _flags(path: Path):
    """Each `add_argument` call of a script: its long options and keyword names."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", "") == "add_argument"
        ):
            options = [
                arg.value
                for arg in node.args
                if isinstance(arg, ast.Constant) and str(arg.value).startswith("--")
            ]
            if options:
                yield options, {keyword.arg for keyword in node.keywords}


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda path: path.name)
def test_a_script_uses_none_of_the_retired_flag_names(path):
    used = {option for options, _ in _flags(path) for option in options}
    assert {old: RENAMED[old] for old in used if old in RENAMED} == {}


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda path: path.name)
def test_every_flag_has_a_help_text(path):
    missing = [
        options[0] for options, keywords in _flags(path) if "help" not in keywords
    ]
    assert missing == []


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda path: path.name)
def test_a_script_builds_its_parser_with_new_parser(path):
    source = path.read_text(encoding="utf-8")
    assert "argparse.ArgumentParser(" not in source or path.name == "_cli.py"
