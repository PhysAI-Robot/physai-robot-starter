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
    }
