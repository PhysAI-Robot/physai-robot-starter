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


def test_isaac_env_config_coerces_manifest_dicts():
    from physai.robots.so101.isaac_env import (
        FrontCameraConfig,
        GraspCubeConfig,
        IsaacEnvConfig,
        TableConfig,
    )

    cfg = IsaacEnvConfig(
        cameras=["front", "wrist"],
        cube={"mass": 0.05},
        front_camera={},
        table={"friction": 0.9},
        target_pos=[0.2, -0.1, 0.021],
    )

    assert cfg.cameras == ("front", "wrist")
    assert cfg.cube == GraspCubeConfig(mass=0.05)
    assert cfg.front_camera == FrontCameraConfig()
    assert cfg.table == TableConfig(friction=0.9)
    assert cfg.target_pos == (0.2, -0.1, 0.021)


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
