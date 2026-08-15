"""Write the composed scene to a single MJCF file.

Useful for opening the task in the stock MuJoCo viewer, or for handing the
scene to another tool:

    python scripts/export_scene.py
    python -m mujoco.viewer --mjcf=outputs/scene_pick_place.xml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from physai.sim import export_xml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("outputs/scene_pick_place.xml"))
    args = ap.parse_args()
    # Must land beside the robot XML so the relative meshdir still resolves.
    path = export_xml(args.out)
    print(f"wrote {path}")
    print("note: mesh paths are relative to assets/so101/, so copy the file "
          "there before loading it standalone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
