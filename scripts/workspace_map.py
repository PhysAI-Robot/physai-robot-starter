"""Print the region of the table where a top-down grasp is actually reachable.

The SO-101 has 5 arm DoF and no shoulder roll, so "the point is reachable" and
"the point is reachable with the jaws pointing down" are different questions.
Run this before widening the cube randomisation range, or the scripted expert
will fail on cubes it physically cannot grasp — which looks like a bad policy.

    python scripts/workspace_map.py
    python scripts/workspace_map.py --hover 0.06 --z 0.034
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np

from physai.robots.so101 import EnvConfig, SO101Env
from physai.robots.so101.kinematics import TOP_DOWN


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", type=float, default=0.034, help="object centre height")
    ap.add_argument("--hover", type=float, default=0.045, help="pre-grasp clearance")
    ap.add_argument("--x-range", type=float, nargs=2, default=(0.14, 0.30))
    ap.add_argument("--y-range", type=float, nargs=2, default=(-0.14, 0.14))
    ap.add_argument("--step", type=float, default=0.01)
    args = ap.parse_args()

    env = SO101Env(EnvConfig(seed=0, render=False))
    q0 = env.reset().joint_state.position[:5]

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
            grasp = env.kin.ik_pinch(np.array([x, y, args.z]), TOP_DOWN, q_init=q0)
            hover = env.kin.ik_pinch(np.array([x, y, args.z + args.hover]),
                                     TOP_DOWN, q_init=q0)
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
