import time

import pytest
from conftest import requires_turtlebot_assets

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.assets,
    pytest.mark.slow,
    requires_turtlebot_assets,
]


def test_turtlebot_host_keeps_ticking_under_the_web_host():
    from physai.robots import create_robot
    from physai.robots.turtlebot import TurtleBot4Config
    from physai.web.host import Host

    robot = create_robot("turtlebot4", config=TurtleBot4Config(max_steps=50))
    host = Host.for_robot(robot, robot_name="turtlebot4", reset_seed=0)
    host.start()
    try:
        deadline = time.monotonic() + 10.0
        steps = 0
        while time.monotonic() < deadline and steps < 5:
            state = host.latest_state()
            steps = state["step"] if state else 0
            time.sleep(0.05)

        assert steps >= 5
        assert host._thread.is_alive()
    finally:
        host.stop()
