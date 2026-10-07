"""Print the region of the table where a top-down grasp is actually reachable.

The SO-101 has 5 arm DoF and no shoulder roll, so "the point is reachable" and
"the point is reachable with the jaws pointing down" are different questions.
Run this before widening the cube randomisation range, or the scripted expert
will fail on cubes it physically cannot grasp — which looks like a bad policy.

    python scripts/workspace_map.py
    python scripts/workspace_map.py --hover 0.06 --z 0.034
    python scripts/workspace_map.py --tilt 60 --hover 0.05

`--tilt` is the gripper's angle below horizontal, pointing away from the base
(90 = top-down, the default; 0 = flat). The hover point is `--hover` back
along the approach direction. The IK check has no collision or contact test.
"""

from __future__ import annotations


import _bootstrap  # noqa: F401
from _cli import new_parser
import numpy as np

from physai.robots.so101 import EnvConfig, SO101Env


def main() -> int:
    ap = new_parser(__doc__)
    ap.add_argument("--z", type=float, default=0.034, help="object centre height")
    ap.add_argument("--hover", type=float, default=0.045, help="pre-grasp clearance")
    ap.add_argument(
        "--tilt",
        type=float,
        default=90.0,
        help="gripper angle below horizontal in degrees (90 = top-down)",
    )
    ap.add_argument(
        "--x-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(0.14, 0.30),
        help="x range scanned, in metres",
    )
    ap.add_argument(
        "--y-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(-0.14, 0.14),
        help="y range scanned, in metres",
    )
    ap.add_argument(
        "--step",
        type=float,
        default=0.01,
        help="grid spacing along x in metres (y uses 0.02)",
    )
    args = ap.parse_args()

    env = SO101Env(EnvConfig(seed=0, render=False))
    q0 = env.reset().joint_state.position[:5]

    tilt = np.radians(args.tilt)

    def approach(x: float, y: float) -> np.ndarray:
        phi = np.arctan2(y, x)
        return np.array(
            [np.cos(phi) * np.cos(tilt), np.sin(phi) * np.cos(tilt), -np.sin(tilt)]
        )

    xs = np.arange(args.x_range[0], args.x_range[1] + 1e-9, args.step)
    ys = np.arange(args.y_range[0], args.y_range[1] + 1e-9, 0.02)

    print("o = grasp + hover reachable top-down   . = grasp only   (blank) = neither\n")
    print("  x  |" + "".join(f"{y:+6.2f}" for y in ys))
    print("-----+" + "-" * (6 * len(ys)))

    good_x = []
    for x in xs:
        cells = []
        n_ok = 0
        for y in ys:
            direction = approach(x, y)
            point = np.array([x, y, args.z])
            grasp = env.kin.ik_pinch(point, direction, q_init=q0)
            hover = env.kin.ik_pinch(
                point - args.hover * direction, direction, q_init=q0
            )
            if grasp.converged and hover.converged:
                cells.append("o")
                n_ok += 1
            elif grasp.converged:
                cells.append(".")
            else:
                cells.append(" ")
        if n_ok:
            good_x.append(x)
        print(f"{x:.2f} |" + "".join(f"{c:>6}" for c in cells))

    env.close()
    if good_x:
        print(f"\nfully graspable x band: [{min(good_x):.2f}, {max(good_x):.2f}]")
    else:
        print("\nno fully graspable cells — check --z and --hover")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
