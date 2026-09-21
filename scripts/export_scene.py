"""Write the composed scene to a single MJCF file.

Useful for opening the task in the stock MuJoCo viewer, or for handing the
scene to another tool:

    python scripts/export_scene.py
    python scripts/export_scene.py --scene sorting_minimal
    python -m mujoco.viewer --mjcf=outputs/scene_pick_place.xml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from physai.robots.registry import scene_defaults
from physai.sim import available_scenes, create_scene, export_xml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("outputs/scene_pick_place.xml"))
    ap.add_argument("--robot", default="so101")
    ap.add_argument("--scene", default="pick_place_minimal", choices=available_scenes())
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
