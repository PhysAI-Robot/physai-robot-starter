"""SO-101 defaults for generic manipulation scene primitives."""

from __future__ import annotations

from ...sim.scenes.common import REPO_ROOT


def scene_defaults() -> dict[str, object]:
    """Return the SO-101 model and end-effector attachment configuration."""
    model_name = "so101_new_calib_camera.xml"
    model_path = REPO_ROOT / "assets" / "so101" / model_name
    if not model_path.exists():
        model_path = REPO_ROOT / "assets" / "so101" / "so101_new_calib.xml"
    return {
        "robot_xml": model_path,
        "ee_site": "gripperframe",
        "gripper_joint": "gripper",
        "static_pad_body": "gripper",
        "moving_pad_body": "moving_jaw_so101_v1",
        "wrist_body": "wrist",
        # The grasp pads are collision boxes standing in for the finger meshes.
        # They are fitted to the SO-101 fingertips: each pad's outer face is
        # flush with the tip's inner face, its 12 x 12 mm footprint is the
        # bounding square of the tapered tip face and sits on the finger centre
        # line, and it is 6 mm thick so a squeezed cube cannot pass through it
        # into the finger. Only the last ~6 mm of each fingertip is a flat face
        # (behind it the lattice is recessed), so the pads are fitted to that
        # zone. Narrower pads (8-10 mm) fit the tip more tightly but make
        # `visual_servo` miss the cube on some seeds: it grasps up to ~15 mm off
        # the pinch centre.
        "pad_size": (0.006, 0.006, 0.003),
        # The gripper angle (rad) at which the pad faces are one cube width
        # (28 mm) apart at the pad centres.
        "pad_align_gripper_q": 0.16,
        "static_pad_pos": (-0.0109, -0.0002, -0.0979),
        "moving_pad_pos": (-0.0093, -0.0753, 0.0190),
        # Rotation of the moving pad about its lateral axis, on top of being
        # parallel to the static pad at `pad_align_gripper_q`. The moving
        # finger's face is not parallel to the static one (the fingers form a V,
        # about 8 degrees apart), so this lays the pad along that face.
        # Positive raises the face toward the tip.
        "moving_pad_tilt": 0.1348,
        # The wrist camera looks along -z of its own frame. With x = (-1, 0, 0)
        # the derived view direction pointed backwards and up, away from the
        # workspace, so it rendered a black frame for the whole episode.
        # Negating the x axis flips the view onto the jaws and the object below
        # them while keeping the original up vector.
        "wrist_cam_pos": (0.0, -0.07, 0.05),
        "wrist_cam_xyaxes": (1.0, 0.0, 0.0, 0.0, 0.7, 0.7),
    }
