import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ASSETS = (
    Path(__file__).resolve().parents[1] / "assets" / "so101" / "so101_new_calib.xml"
)

requires_assets = pytest.mark.skipif(
    not ASSETS.exists(),
    reason="run `python scripts/fetch_assets.py` to download the SO-101 description",
)

TURTLEBOT_ASSETS = (
    Path(__file__).resolve().parents[1] / "assets" / "turtlebot4" / "turtlebot4.xml"
)

requires_turtlebot_assets = pytest.mark.skipif(
    not TURTLEBOT_ASSETS.exists(),
    reason="run `python scripts/fetch_assets.py --robot turtlebot4`",
)


VIPERX300S_ASSETS = (
    Path(__file__).resolve().parents[1] / "assets" / "viperx300s" / "vx300s.xml"
)

requires_viperx300s_assets = pytest.mark.skipif(
    not VIPERX300S_ASSETS.exists(),
    reason="run `python scripts/fetch_assets.py --robot viperx300s`",
)


@pytest.fixture
def env():
    from physai.robots.so101 import EnvConfig, SO101Env
    from physai.tasks import TaskRuntime, create_task

    robot = SO101Env(EnvConfig(seed=0, render=False, max_steps=200))
    e = TaskRuntime(robot, create_task("single_cube_place"))
    yield e
    e.close()
