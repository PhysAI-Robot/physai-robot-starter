"""Write a manifest's composed scene to a single MJCF file.

Useful for opening the task in the stock MuJoCo viewer, or for handing the
scene to another tool:

    python scripts/export_scene.py
    python scripts/export_scene.py --manifest configs/manifests/so101_sorting.yaml
    python -m mujoco.viewer --mjcf=outputs/scene_so101_single_cube_fixed_place.xml
"""

from __future__ import annotations

from pathlib import Path

import _bootstrap  # noqa: F401
from _cli import new_parser
from _common_args import add_manifest, add_out

from physai.config import load_manifest
from physai.runtime import robot_env_config
from physai.sim.mujoco import export_xml


def main() -> int:
    ap = new_parser(__doc__)
    add_manifest(
        ap,
        help="session manifest whose scene is exported: its robot's scene defaults "
        "and its scene overrides",
    )
    add_out(
        ap,
        default=None,
        help="MJCF file to write (default: outputs/scene_<manifest name>.xml)",
    )
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    if manifest.world is not None or len(manifest.robots) != 1:
        ap.error("--manifest must describe exactly one robot")
    robot = manifest.robots[0].robot
    scene = getattr(robot_env_config(manifest), "scene", None)
    if scene is None:
        ap.error(f"robot {robot!r} has no scene to export")
    out = args.out or Path("outputs") / f"scene_{args.manifest.stem}.xml"
    # Must land beside the robot XML so the relative meshdir still resolves.
    path = export_xml(out, scene)
    print(f"wrote {path}")
    print(
        f"note: mesh paths are relative to assets/{robot}/, so copy the "
        "file there before loading it standalone."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
