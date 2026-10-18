"""Why is the ViperX 300 S arm missing from Isaac camera frames? Render one front frame under
one configuration and save it, so configurations can be compared by eye.

    python scripts/diag_viperx300s_render.py --variant none
    python scripts/diag_viperx300s_render.py --variant physx --gravcomp

Writes outputs/viperx300s/isaac/diag_<variant>_<gc|nogc>.png and prints what the stage holds.
"""

from __future__ import annotations

from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
from _cli import new_parser

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "viperx300s" / "isaac"


def main() -> int:
    parser = new_parser(__doc__)
    parser.add_argument(
        "--variant",
        default="physx",
        help="USD Physics variant to import (none, physx, ...)",
    )
    parser.add_argument(
        "--gravcomp", action="store_true", help="switch gravity off on the arm's links"
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    from physai.robots.description import load_robot_description
    from physai.robots.viperx300s.isaac_env import _DESCRIPTION_PATH
    from physai.robots.viperx300s.scene import FRONT_CAM_POS, FRONT_CAM_XYAXES
    from physai.sim.isaac.core import IsaacSimulationCore, ensure_simulation_app
    from physai.sim.workspace import FRONT_CAMERA_FOVY_DEG

    ensure_simulation_app(headless=True)
    from physai.sim.isaac import description as isaac_desc
    from physai.sim.isaac import scene as isaac_scene

    desc = load_robot_description(_DESCRIPTION_PATH)
    _usd, robot = isaac_desc.import_robot(
        desc,
        REPO / "assets" / "viperx300s",
        usd_out_dir=REPO / ".isaac_cache" / "viperx300s",
        variant=args.variant,
    )
    stage = isaac_desc.get_stage()
    isaac_desc.ensure_physics_scene(stage)
    if args.gravcomp:
        print("gravcomp links:", isaac_desc.apply_gravity_compensation(stage, robot))
    isaac_scene.add_studio_lighting(stage, scale=1.0)
    isaac_scene.add_ground_plane(
        stage,
        texture_path=isaac_scene.checker_texture(
            REPO / ".isaac_cache" / "studio" / "floor_checker.png"
        ),
    )
    cam = isaac_scene.add_world_camera(
        stage,
        "/World_front_camera",
        position=FRONT_CAM_POS,
        quat_wxyz=isaac_scene.xyaxes_to_quat_wxyz(FRONT_CAM_XYAXES),
        fovy_deg=FRONT_CAMERA_FOVY_DEG,
        width=320,
        height=240,
    )
    core = IsaacSimulationCore(
        control_hz=30.0,
        physics_hz=60.0,
        render=True,
        camera_width=320,
        camera_height=240,
    )
    core.reset_simulation()
    for _ in range(10):
        core.step_simulation()
    for _ in range(4):
        core.render_camera(cam)
    frame = core.render_camera(cam)

    from PIL import Image

    name = f"diag_{args.variant}_{'gc' if args.gravcomp else 'nogc'}"
    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(OUT / f"{name}.png")
    print(name, "mean pixel", float(frame.mean()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
