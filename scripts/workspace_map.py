"""Print the region of the table where a grasp is actually reachable.

A point being reachable and the point being reachable with the jaws pointing down
are different questions, and the second one is what a scripted expert needs. Run this
before widening the cube randomisation range, or the expert will fail on cubes it
physically cannot grasp, which looks like a bad policy.

    python scripts/workspace_map.py
    python scripts/workspace_map.py --manifest configs/manifests/viperx300s_single_cube_fixed_place.yaml
    python scripts/workspace_map.py --hover 0.06 --z 0.034
    python scripts/workspace_map.py --tilt 60 --hover 0.05

The robot of the manifest answers each cell through its own `grasp_reach`, so the grasp
(which way the gripper points, at what height, with what hover and carry) is the one that
robot's expert asks for; `--z`, `--hover`, `--carry` and `--tilt` override its defaults and
a robot that cannot take one says so. A cell is `o` only when the grasp, the hover above it
and the carry lift are all reachable. The IK check has no collision or contact test.

By default the scan covers the table top of the manifest's scene. After the grid, a summary
prints the distance from the base of the `o` cells and their y extent, which is what a
`spawn_radius_range` and an |y| bound are taken from.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
from _common_args import add_manifest, open_mujoco_session
from _cli import new_parser
import numpy as np


def main() -> int:
    ap = new_parser(__doc__)
    add_manifest(ap, help="session manifest whose robot and table are scanned")
    ap.add_argument(
        "--z",
        type=float,
        default=None,
        help="grasp height in m (default: the robot's own, from the table and cube)",
    )
    ap.add_argument(
        "--hover",
        type=float,
        default=None,
        help="pre-grasp clearance in m (default: the robot's own)",
    )
    ap.add_argument(
        "--carry",
        type=float,
        default=None,
        help="extra lift above the grasp that must also be reachable, in m, 0 for "
        "none (default: the robot's own)",
    )
    ap.add_argument(
        "--tilt",
        type=float,
        default=90.0,
        help="gripper angle below horizontal in degrees, pointing away from the "
        "base (90 = top-down); a robot that only grasps top-down refuses others",
    )
    ap.add_argument(
        "--x-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="x range scanned, in metres (default: the table top)",
    )
    ap.add_argument(
        "--y-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="y range scanned, in metres (default: the table top)",
    )
    ap.add_argument(
        "--step", type=float, default=0.01, help="grid spacing along x in metres"
    )
    ap.add_argument(
        "--y-step", type=float, default=0.02, help="grid spacing along y in metres"
    )
    args = ap.parse_args()

    # IK is the same on every engine, so the cheapest one answers.
    session = open_mujoco_session(args.manifest)
    env = session.runtime.robot
    if not hasattr(env, "grasp_reach"):
        ap.error(f"the manifest's robot has no grasp_reach to scan: {args.manifest}")
    q0 = env.reset().joint_state.position

    defaults = env.grasp_reach_defaults()
    z = defaults["z"] if args.z is None else args.z
    hover = defaults["hover"] if args.hover is None else args.hover
    carry = defaults["carry"] if args.carry is None else args.carry
    pos, size = env.cfg.scene.table_pos, env.cfg.scene.table_size
    x_lo, x_hi = args.x_range or (pos[0] - size[0], pos[0] + size[0])
    y_lo, y_hi = args.y_range or (pos[1] - size[1], pos[1] + size[1])
    xs = np.arange(x_lo, x_hi + 1e-9, args.step)
    ys = np.arange(y_lo, y_hi + 1e-9, args.y_step)

    print("o = grasp + hover reachable   . = grasp only   (blank) = neither")
    if carry > 0:
        print(f"(o also needs the carry point, {carry:.3f} m above the grasp)")
    print(f"grasp z = {z:.3f} m, hover {hover:.3f} m, tilt {args.tilt:.0f} deg\n")
    print("  x  |" + "".join(f"{y:+6.2f}" for y in ys))
    print("-----+" + "-" * (6 * len(ys)))

    good_x = []
    good_cells = []
    for x in xs:
        cells = []
        for y in ys:
            try:
                reach = env.grasp_reach(
                    x, y, z=z, hover=hover, carry=carry, tilt=args.tilt, q_init=q0
                )
            except ValueError as exc:
                ap.error(str(exc))
            if reach.ok:
                cells.append("o")
                good_cells.append((x, y))
            else:
                cells.append("." if reach.grasp else " ")
        if "o" in cells:
            good_x.append(x)
        print(f"{x:.2f} |" + "".join(f"{c:>6}" for c in cells))

    session.close()
    if good_x:
        pts = np.array(good_cells)
        radius = np.hypot(pts[:, 0], pts[:, 1])
        print(f"\nfully graspable x band: [{min(good_x):.2f}, {max(good_x):.2f}]")
        print(
            f"{len(pts)} cells of {len(xs) * len(ys)}; distance from base "
            f"{radius.min():.3f}-{radius.max():.3f} m; y {pts[:, 1].min():+.2f} "
            f"to {pts[:, 1].max():+.2f} m"
        )
    else:
        print("\nno fully graspable cells — check --z and --hover")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
