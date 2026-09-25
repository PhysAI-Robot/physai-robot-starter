"""`ArmKinematics.tool_pose`: the pinch centre, oriented relative to top-down."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from physai.robots import create_robot
from physai.robots.so101 import TOP_DOWN, EnvConfig

MODEL = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "so101"
    / "so101_new_calib_camera.xml"
)

pytestmark = [
    pytest.mark.assets,
    pytest.mark.skipif(
        not MODEL.exists(),
        reason="run `python scripts/fetch_assets.py` to download the camera variant",
    ),
]


@pytest.fixture(scope="module")
def robot():
    env = create_robot("so101", config=EnvConfig(render=False))
    env.reset(seed=0)
    return env


def rpy_deg(pose):
    """Intrinsic ZYX roll/pitch/yaw, the same convention the browser shows."""
    x, y, z, w = pose.pose.orientation.as_array()
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.degrees([roll, pitch, yaw])


def place_top_down(robot, target, pan=0.0):
    """Pose the arm for a top-down grasp at `target`, then pan by `pan` rad."""
    result = robot.kin.ik_pinch(np.asarray(target), TOP_DOWN)
    assert result.position_error < 2e-3
    qpos = result.qpos.copy()
    qpos[0] += pan
    robot.data.qpos[robot.kin.qpos_adr] = qpos
    mujoco.mj_forward(robot.model, robot.data)


def test_position_is_the_pinch_centre(robot):
    place_top_down(robot, [0.22, 0.0, 0.05])

    pose = robot.kin.tool_pose(robot.data)

    np.testing.assert_allclose(
        pose.pose.position.as_array(), robot.kin.pinch_center(robot.data)
    )
    assert pose.header.frame_id == "base"


def test_a_top_down_grasp_reads_near_zero_and_is_far_from_gimbal_lock(robot):
    for target in ([0.22, 0.0, 0.05], [0.18, 0.05, 0.04], [0.25, -0.06, 0.06]):
        place_top_down(robot, target)

        roll, pitch, _ = rpy_deg(robot.kin.tool_pose(robot.data))

        assert abs(roll) < 5 and abs(pitch) < 5, target


def test_panning_the_arm_changes_only_yaw(robot):
    place_top_down(robot, [0.22, 0.0, 0.05])
    _, _, yaw0 = rpy_deg(robot.kin.tool_pose(robot.data))

    for pan_deg in (-40.0, 25.0):
        place_top_down(robot, [0.22, 0.0, 0.05], pan=np.radians(pan_deg))

        roll, pitch, yaw = rpy_deg(robot.kin.tool_pose(robot.data))

        assert abs(roll) < 5 and abs(pitch) < 5
        # shoulder_pan is positive clockwise seen from above, yaw counter-clockwise.
        assert yaw - yaw0 == pytest.approx(-pan_deg, abs=1)


def test_the_site_frame_itself_is_singular_at_top_down(robot):
    """Why tool_pose exists: the raw site orientation is at gimbal lock here."""
    place_top_down(robot, [0.22, 0.0, 0.05])

    site_pitch = rpy_deg(robot.kin.fk(robot.data))[1]

    assert abs(site_pitch) > 80
