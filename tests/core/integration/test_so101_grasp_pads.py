"""The SO-101 grasp pads are fitted to the fingertips (see ManipulationSceneConfig)."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from physai.robots import create_robot
from physai.robots.so101 import EnvConfig
from physai.sim.mujoco import SingleCubeFixedPlaceSceneConfig

MODEL = (
    Path(__file__).resolve().parents[3]
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
    scene = SingleCubeFixedPlaceSceneConfig()
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
    derivation = scene.description.derivation
    close_to(robot, derivation["pad_align_gripper_q"])

    static_centre, static_axes, half = pad_frame(robot, "pad_static")
    moving_centre, moving_axes, _ = pad_frame(robot, "pad_moving")

    # The static pad lies parallel to the gripper's pinch axis.
    assert abs(static_axes[2, 2]) == pytest.approx(1.0, abs=1e-3)
    # The moving pad is tilted along its finger's face, by exactly the tilt.
    tilt = np.arctan2(moving_axes[0, 2], moving_axes[2, 2])
    assert abs(tilt) == pytest.approx(derivation["moving_pad_tilt"], abs=1e-3)
    # Face to face at the pad centres, the gap is the cube width.
    gap = (moving_centre[2] - half[2]) - (static_centre[2] + half[2])
    assert gap == pytest.approx(2 * scene.cube_half, abs=0.5 * MM)


def test_baked_pad_quaternions_match_their_forward_kinematics_derivation(robot):
    """`description.yaml`'s `contact_pads[].quat` are baked FK output, not
    hand-picked (see its `derivation` block and `sim.scenes.common
    .apply_description`, which only ever reads the baked values). Re-deriving
    them here — independently of the scene builder — keeps the fit
    traceable: a future re-tune edits `derivation` and re-runs this test to
    get the new quaternions to paste back into the YAML.
    """
    desc = robot.cfg.scene.description
    derivation = desc.derivation
    model = mujoco.MjModel.from_xml_path(str(robot.cfg.scene.robot_xml))
    data = mujoco.MjData(model)
    grip_jid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, derivation["gripper_joint"]
    )
    data.qpos[model.jnt_qposadr[grip_jid]] = derivation["pad_align_gripper_q"]
    mujoco.mj_forward(model, data)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, desc.ee_site)
    site_rotation = data.site_xmat[site_id].reshape(3, 3)
    pads_by_name = {pad.name: pad for pad in desc.contact_pads}
    for name, is_moving in (("pad_static", False), ("pad_moving", True)):
        pad = pads_by_name[name]
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, pad.parent_link)
        local_rotation = data.xmat[body_id].reshape(3, 3).T @ site_rotation
        if is_moving:
            tilt = derivation["moving_pad_tilt"]
            c, s = np.cos(tilt), np.sin(tilt)
            local_rotation = local_rotation @ np.array(
                [[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]]
            )
        derived = np.zeros(4)
        mujoco.mju_mat2Quat(derived, local_rotation.reshape(9))
        np.testing.assert_allclose(derived, pad.quat, atol=1e-6)


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
    close_to(robot, robot.cfg.scene.description.derivation["pad_align_gripper_q"])

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


def test_pads_sit_at_the_tip_and_the_grasp_torque_is_capped(robot):
    close_to(robot, robot.cfg.scene.description.derivation["pad_align_gripper_q"])

    for name, body in (
        ("pad_static", "gripper"),
        ("pad_moving", "moving_jaw_so101_v1"),
    ):
        centre, _, half = pad_frame(robot, name)
        tip = finger_tip_x(robot, body)

        assert abs(centre[1]) < 0.5 * MM  # on the finger centre line
        assert centre[0] + half[0] <= tip  # never past the fingertip
        assert centre[0] + half[0] >= tip - 1.5 * MM  # and reaching right up to it

    limit = robot.model.actuator_forcerange[robot.grip_act_id]
    np.testing.assert_allclose(limit, [-0.3, 0.3])  # not the model's own rating
