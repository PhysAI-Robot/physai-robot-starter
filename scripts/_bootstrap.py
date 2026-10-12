"""Make `physai` and your own plugin folders importable when running scripts
from a source checkout.

Avoids forcing `uv sync` before the first run. A plugin folder such as
`my_lab/` placed next to `src/` is not part of the installed package, so the
repository root goes on `sys.path` the same way `src/` does for `physai`; a
manifest then names its module in `plugins:` (see physai.plugins).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (SRC, ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
