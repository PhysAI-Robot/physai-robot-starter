"""The SO-101 grasp pads are fitted to the fingertips (see ManipulationSceneConfig)."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from physai.robots import create_robot
from physai.robots.so101 import EnvConfig
from physai.sim import PickPlaceMinimalSceneConfig

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

MM = 1e-3


@pytest.fixture(scope="module")
def robot():
    scene = PickPlaceMinimalSceneConfig()
    env = create_robot("so101", config=EnvConfig(scene=scene, render=False))
    env.reset(seed=0)
    return env


def pad_frame(robot, name):
    """A pad's centre and half-size in the site frame (x approach, y lateral, z pinch)."""
    data, model, kin = robot.data, robot.model, robot.kin
    site_rotation = data.site_xmat[kin.site_id].reshape(3, 3)
    geom = model.geom(name).id
    centre = (data.geom_xpos[geom] - data.site_xpos[kin.site_id]) @ site_rotation
    axes = site_rotation.T @ data.geom_xmat[geom].reshape(3, 3)
    return centre, axes, model.geom_size[geom]


def close_to(robot, angle):
    robot.data.qpos[robot.grip_qadr] = angle
    mujoco.mj_forward(robot.model, robot.data)


def test_pad_faces_are_one_cube_apart_at_the_alignment_angle(robot):
    scene = robot.cfg.scene
    close_to(robot, scene.pad_align_gripper_q)

    static_centre, static_axes, half = pad_frame(robot, "pad_static")
    moving_centre, moving_axes, _ = pad_frame(robot, "pad_moving")

    # The static pad lies parallel to the gripper's pinch axis.
    assert abs(static_axes[2, 2]) == pytest.approx(1.0, abs=1e-3)
    # The moving pad is tilted along its finger's face, by exactly the tilt.
    tilt = np.arctan2(moving_axes[0, 2], moving_axes[2, 2])
    assert abs(tilt) == pytest.approx(scene.moving_pad_tilt, abs=1e-3)
    # Face to face at the pad centres, the gap is the cube width.
    gap = (moving_centre[2] - half[2]) - (static_centre[2] + half[2])
    assert gap == pytest.approx(2 * scene.cube_half, abs=0.5 * MM)


def flat_face_slope(robot, body, mesh_prefix, toward_cube):
    """Slope dz/dx of a finger's flat tip face, in the site frame.

    The inner surface is sampled in 1 mm slices anchored at the fingertip; only
    the last few millimetres are flat, behind them the lattice is recessed.
    """
    model, data, kin = robot.model, robot.data, robot.kin
    site_rotation = data.site_xmat[kin.site_id].reshape(3, 3)
    for geom in range(model.ngeom):
        if (
            model.geom_bodyid[geom] == model.body(body).id
            and model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH
            and model.geom_group[geom] == 2
            and model.mesh(int(model.geom_dataid[geom])).name.startswith(mesh_prefix)
        ):
            mesh = int(model.geom_dataid[geom])
            start = int(model.mesh_vertadr[mesh])
            local = model.mesh_vert[start : start + int(model.mesh_vertnum[mesh])]
            world = data.geom_xpos[geom] + local @ data.geom_xmat[geom].reshape(3, 3).T
            v = (world - data.site_xpos[kin.site_id]) @ site_rotation
            v = v[np.abs(v[:, 1]) < 4 * MM]
            tip = v[:, 0].max()
            xs, zs = [], []
            for k in range(6):
                sl = v[(v[:, 0] <= tip - k * MM) & (v[:, 0] > tip - (k + 1) * MM)]
                xs.append(sl[:, 0].mean())
                zs.append(sl[:, 2].max() if toward_cube > 0 else sl[:, 2].min())
            return float(np.polyfit(xs, zs, 1)[0])
    raise KeyError(body)


def test_each_pad_lies_along_its_fingers_flat_tip_face(robot):
    close_to(robot, robot.cfg.scene.pad_align_gripper_q)

    for name, body, prefix, toward_cube in (
        ("pad_static", "gripper", "wrist_roll_follower", +1),
        ("pad_moving", "moving_jaw_so101_v1", "moving_jaw", -1),
    ):
        _, axes, _ = pad_frame(robot, name)
        pad_slope = -axes[0, 2] / axes[2, 2]
        finger_slope = flat_face_slope(robot, body, prefix, toward_cube)

        # within one degree of the finger's own face
        assert np.degrees(abs(np.arctan(pad_slope) - np.arctan(finger_slope))) < 1.0


def finger_tip_x(robot, body):
    """The finger mesh's furthest point along the approach axis, in the site frame."""
    model, data, kin = robot.model, robot.data, robot.kin
    site_rotation = data.site_xmat[kin.site_id].reshape(3, 3)
    for geom in range(model.ngeom):
        if (
            model.geom_bodyid[geom] == model.body(body).id
            and model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH
            and model.geom_group[geom] == 2
            and model.mesh(int(model.geom_dataid[geom])).name.startswith(
                ("wrist_roll_follower", "moving_jaw")
            )
        ):
            mesh = int(model.geom_dataid[geom])
            start = int(model.mesh_vertadr[mesh])
            local = model.mesh_vert[start : start + int(model.mesh_vertnum[mesh])]
            world = data.geom_xpos[geom] + local @ data.geom_xmat[geom].reshape(3, 3).T
            return float(
                ((world - data.site_xpos[kin.site_id]) @ site_rotation)[:, 0].max()
            )
    raise KeyError(body)


def test_pads_sit_on_the_finger_centre_line_at_the_tip(robot):
    close_to(robot, robot.cfg.scene.pad_align_gripper_q)

    for name, body in (
        ("pad_static", "gripper"),
        ("pad_moving", "moving_jaw_so101_v1"),
    ):
        centre, _, half = pad_frame(robot, name)
        tip = finger_tip_x(robot, body)

        assert abs(centre[1]) < 0.5 * MM  # on the finger centre line
        assert centre[0] + half[0] <= tip  # never past the fingertip
        assert centre[0] + half[0] >= tip - 1.5 * MM  # and reaching right up to it


def test_the_grasp_torque_cap_is_not_the_models_own_rating(robot):
    limit = robot.model.actuator_forcerange[robot.grip_act_id]

    np.testing.assert_allclose(limit, [-0.3, 0.3])
