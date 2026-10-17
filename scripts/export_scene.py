"""Write the composed scene to a single MJCF file.

Useful for opening the task in the stock MuJoCo viewer, or for handing the
scene to another tool:

    python scripts/export_scene.py
    python scripts/export_scene.py --scene sorting_minimal
    python -m mujoco.viewer --mjcf=outputs/scene_single_cube_place.xml
"""

from __future__ import annotations

from pathlib import Path

import _bootstrap  # noqa: F401
from _cli import new_parser
from _common_args import add_out, add_robot

from physai.robots.registry import scene_defaults
from physai.sim.mujoco import available_scenes, create_scene, export_xml


def main() -> int:
    ap = new_parser(__doc__)
    add_out(
        ap,
        default=Path("outputs/scene_single_cube_place.xml"),
        help="MJCF file to write",
    )
    add_robot(ap, default="so101")
    ap.add_argument(
        "--scene",
        default="single_cube_place",
        choices=available_scenes(),
        help="registered scene to export",
    )
    args = ap.parse_args()

    cfg = create_scene(args.scene, **scene_defaults(args.robot))
    # Must land beside the robot XML so the relative meshdir still resolves.
    path = export_xml(args.out, cfg)
    print(f"wrote {path}")
    print(
        f"note: mesh paths are relative to assets/{args.robot}/, so copy the "
        "file there before loading it standalone."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
