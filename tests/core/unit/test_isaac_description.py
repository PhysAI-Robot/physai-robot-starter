"""physai.isaac.description's pure functions: no isaacsim/omni/pxr import,
so these run in the normal test suite (see the import-linter contract
confining those to physai.isaac and robots.so101.isaac_env)."""

import math

import pytest


def test_urdf_link_name_suffixes_the_mjcf_body_name():
    from physai.isaac.description import urdf_link_name

    assert urdf_link_name("gripper") == "gripper_link"
    assert urdf_link_name("wrist_camera") == "wrist_camera_link"


def test_fovy_to_focal_length_matches_the_pinhole_relation():
    from physai.isaac.description import fovy_to_focal_length

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
    from physai.isaac.description import fovy_to_focal_length

    with pytest.raises(ValueError, match="fovy_deg"):
        fovy_to_focal_length(0.0)
    with pytest.raises(ValueError, match="fovy_deg"):
        fovy_to_focal_length(180.0)
