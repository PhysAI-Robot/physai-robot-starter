"""physai.sim.isaac.description's pure functions: no isaacsim/omni/pxr import,
so these run in the normal test suite (see the import-linter contract
confining those to physai.sim.isaac and robots.so101.isaac_env)."""

import math

import pytest
from tests.conftest import requires_assets


def test_urdf_link_name_suffixes_the_mjcf_body_name():
    from physai.sim.isaac.description import urdf_link_name

    assert urdf_link_name("gripper") == "gripper_link"
    assert urdf_link_name("wrist_camera") == "wrist_camera_link"


def test_fovy_to_focal_length_matches_the_pinhole_relation():
    from physai.sim.isaac.description import fovy_to_focal_length

    # A 90-degree vertical FOV means half-FOV is 45 degrees, so the pinhole
    # relation (aperture = 2 * focal_length * tan(fovy / 2)) reduces to
    # focal_length == aperture / 2.
    aperture = 20.955
    assert fovy_to_focal_length(90.0, vertical_aperture_mm=aperture) == pytest.approx(
        aperture / 2.0
    )

    # A narrower FOV needs a longer focal length (more "zoomed in").
    narrow = fovy_to_focal_length(30.0, vertical_aperture_mm=aperture)
    wide = fovy_to_focal_length(90.0, vertical_aperture_mm=aperture)
    assert narrow > wide

    # Round-trips through the same relation the function solves.
    fovy_deg = 62.0
    focal_length = fovy_to_focal_length(fovy_deg, vertical_aperture_mm=aperture)
    recovered = 2.0 * math.degrees(math.atan(aperture / (2.0 * focal_length)))
    assert recovered == pytest.approx(fovy_deg)


def test_fovy_to_focal_length_rejects_a_degenerate_fov():
    from physai.sim.isaac.description import fovy_to_focal_length

    with pytest.raises(ValueError, match="fovy_deg"):
        fovy_to_focal_length(0.0)
    with pytest.raises(ValueError, match="fovy_deg"):
        fovy_to_focal_length(180.0)


@requires_assets
def test_wrist_calibration_follows_the_chosen_resolution():
    """A mounted camera's pixel size is the render resolution, not the
    description's own 320x240 default: using the latter at 640x480 halved
    cx/cy/fy and misplaced the triangulated cube by ~12 cm."""
    import mujoco

    from physai.robots.description import load_robot_description
    from physai.robots.so101.isaac_env import (
        _DESCRIPTION_PATH,
        REPO_ROOT,
        IsaacEnvConfig,
        SO101IsaacEnv,
    )
    from physai.robots.so101.kinematics import ArmKinematics

    description = load_robot_description(_DESCRIPTION_PATH)
    model = mujoco.MjModel.from_xml_path(
        str(REPO_ROOT / "assets" / "so101" / description.mjcf)
    )
    base = SO101IsaacEnv.__new__(SO101IsaacEnv)  # no SimulationApp needed
    base.description, base.model = description, model
    base.data = mujoco.MjData(model)
    base.kin = ArmKinematics(model)
    mujoco.mj_forward(model, base.data)
    base.cfg = IsaacEnvConfig()

    results = {}
    for resolution, (width, height) in {
        "320x240": (320, 240),
        "640x480": (640, 480),
        "1280x720": (1280, 720),
    }.items():
        base._render_size = (width, height)
        intrinsics, _ = base.camera_calibration("wrist")
        results[resolution] = intrinsics
        assert intrinsics.cx == (width - 1) / 2.0
        assert intrinsics.cy == (height - 1) / 2.0
        assert intrinsics.fx == intrinsics.fy  # square pixels at any aspect
    # Same field of view, so the focal length scales with the image height.
    assert results["640x480"].fy == 2 * results["320x240"].fy


def test_the_isaac_camera_orientation_matches_mujoco_for_the_front_camera():
    import mujoco
    import numpy as np

    from physai.sim.isaac.scene import xyaxes_to_quat_wxyz
    from physai.sim.mujoco import SingleCubeFixedPlaceSceneConfig

    xyaxes = SingleCubeFixedPlaceSceneConfig().front_cam_xyaxes
    x_axis, y_axis = np.asarray(xyaxes[:3]), np.asarray(xyaxes[3:])
    x_axis, y_axis = x_axis / np.linalg.norm(x_axis), y_axis / np.linalg.norm(y_axis)
    rotation = np.stack([x_axis, y_axis, np.cross(x_axis, y_axis)], axis=1)
    expected = np.zeros(4)
    mujoco.mju_mat2Quat(expected, rotation.reshape(9))

    got = np.asarray(xyaxes_to_quat_wxyz(xyaxes))

    # q and -q are the same rotation
    assert min(np.abs(got - expected).max(), np.abs(got + expected).max()) < 1e-9


def test_isaac_config_rejects_what_it_cannot_build():
    import pytest

    from physai.robots.so101.isaac_env import IsaacEnvConfig
    from physai.sim.mujoco import (
        SingleCubeFixedPlaceSceneConfig,
        SortingMinimalSceneConfig,
    )

    with pytest.raises(ValueError, match="single-cube"):
        IsaacEnvConfig(scene=SortingMinimalSceneConfig())
    with pytest.raises(ValueError, match="camera_resolution"):
        IsaacEnvConfig(
            scene=SingleCubeFixedPlaceSceneConfig(camera_resolution="640x480"),
        )
    with pytest.raises(ValueError, match="randomize_target"):
        IsaacEnvConfig(randomize_target=True)


def test_the_single_cube_fixed_place_manifest_builds_a_config_for_either_simulator():
    """One manifest describes the environment for both engines, so every robot
    config key it sets must be a field of each env config."""
    from dataclasses import fields

    from physai.config import load_manifest
    from physai.config.compat import with_overrides
    from physai.robots import create_env_config

    manifest = load_manifest("configs/manifests/so101_single_cube_fixed_place.yaml")
    isaac = with_overrides(manifest, simulator="isaac")
    assert isaac.simulator == "isaac"
    accepted = {f.name for f in fields(create_env_config("so101", simulator="isaac"))}
    assert set(isaac.robots[0].config) <= accepted
