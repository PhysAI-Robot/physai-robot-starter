"""Make `physai` importable when running scripts from a source checkout.

Avoids forcing `pip install -e .` before the first run.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
