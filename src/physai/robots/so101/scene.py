"""SO-101 defaults for generic manipulation scene primitives."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]


def scene_defaults() -> dict[str, object]:
    """Return the SO-101 model and end-effector attachment configuration."""
    return {
        "robot_xml": REPO_ROOT / "assets" / "so101" / "so101_new_calib.xml",
        "ee_site": "gripperframe",
        "gripper_joint": "gripper",
        "static_pad_body": "gripper",
        "moving_pad_body": "moving_jaw_so101_v1",
    }
