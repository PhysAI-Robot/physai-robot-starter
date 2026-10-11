"""Make `physai` and `research` importable when running scripts from a
source checkout.

Avoids forcing `uv sync` before the first run. `research/` is a sibling of
`src/`, not part of the installed package, so it needs the repository root
on `sys.path` the same way `physai` needs `src/`.

`research/` is a private git submodule, so a plain checkout may not have it:
`import_research` loads research modules when present and skips them otherwise.
"""

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (SRC, ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def import_research(*modules: str) -> None:
    """Import research modules so they register with the core registries; skip
    them when `research/` is not checked out (a missing dependency of one still raises)."""
    for name in modules:
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if not (exc.name or "").startswith("research"):
                raise
