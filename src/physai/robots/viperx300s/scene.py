"""ViperX 300 S defaults for generic manipulation scene primitives."""

from __future__ import annotations

from pathlib import Path

from ...sim.mujoco.scenes.common import REPO_ROOT
from ...sim.workspace import ViewCamera
from ..description import load_robot_description

_DESCRIPTION_PATH = Path(__file__).resolve().parent / "description.yaml"

# The workspace is laid out for this arm's reach (about 0.7 m, against the SO-101's
# 0.3 m): a table whose near edge is 0.10 m from the base, the cube and the target
# in the middle of it, and a front camera on the far side looking back at the arm.
TABLE_POS = (0.35, 0.0, 0.01)
TABLE_SIZE = (0.25, 0.30, 0.01)
TARGET_POS = (0.32, -0.11, 0.021)
TARGET_RADIUS = 0.04
CUBE_POS = (0.32, 0.10, 0.034)
# A camera 0.90 m in front of the base and 0.52 m up, looking at the table centre.
FRONT_CAM_POS = (0.90, 0.0, 0.52)
FRONT_CAM_XYAXES = (0.0, 1.0, 0.0, -0.68, 0.0, 0.73)
# The `--video` view: the whole arm and the table from a diagonal.
VIDEO_VIEW = ViewCamera(
    lookat=(0.32, 0.0, 0.22), distance=1.35, azimuth=135.0, elevation=-30.0
)


def scene_defaults() -> dict[str, object]:
    """Return the ViperX 300 S model, end-effector attachment, and workspace layout."""
    description = load_robot_description(_DESCRIPTION_PATH)
    return {
        "robot_xml": REPO_ROOT / "assets" / "viperx300s" / description.mjcf,
        "description": description,
        "table_pos": TABLE_POS,
        "table_size": TABLE_SIZE,
        "target_pos": TARGET_POS,
        "target_radius": TARGET_RADIUS,
        "cube_pos": CUBE_POS,
        "front_cam_pos": FRONT_CAM_POS,
        "front_cam_xyaxes": FRONT_CAM_XYAXES,
        "video_view": VIDEO_VIEW,
    }
