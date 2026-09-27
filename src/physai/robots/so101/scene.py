"""SO-101 defaults for generic manipulation scene primitives."""

from __future__ import annotations

from pathlib import Path

from ...sim.mujoco.scenes.common import REPO_ROOT
from ..description import load_robot_description

_DESCRIPTION_PATH = Path(__file__).resolve().parent / "description.yaml"


def scene_defaults() -> dict[str, object]:
    """Return the SO-101 model and end-effector attachment configuration."""
    description = load_robot_description(_DESCRIPTION_PATH)
    model_path = REPO_ROOT / "assets" / "so101" / description.mjcf
    if not model_path.exists():
        # The camera-variant MJCF may not be fetched (e.g. an older
        # `assets/so101/` snapshot); fall back to the plain calibration file.
        # `apply_description`'s frames/cameras/pads all key off a
        # `parent_link` body name, so this fallback needs no data change: a
        # camera or frame whose parent link the fallback model lacks is
        # simply skipped (see `description.yaml`'s two wrist-camera entries).
        model_path = REPO_ROOT / "assets" / "so101" / "so101_new_calib.xml"
    return {
        "robot_xml": model_path,
        "description": description,
    }
