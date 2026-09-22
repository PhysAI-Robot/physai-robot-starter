"""Make `physai` and `research` importable when running scripts from a
source checkout.

Avoids forcing `uv sync` before the first run. `research/` is a sibling of
`src/`, not part of the installed package, so it needs the repository root
on `sys.path` the same way `physai` needs `src/`.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (SRC, ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
