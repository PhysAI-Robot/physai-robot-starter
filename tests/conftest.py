import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "so101" / "so101_new_calib.xml"

requires_assets = pytest.mark.skipif(
    not ASSETS.exists(),
    reason="run `python scripts/fetch_assets.py` to download the SO-101 description",
)


@pytest.fixture(scope="session")
def env():
    from physai.robots.so101 import EnvConfig, SO101Env

    e = SO101Env(EnvConfig(seed=0, render=False, max_steps=200))
    yield e
    e.close()
