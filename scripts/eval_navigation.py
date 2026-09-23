"""Evaluate the registered deterministic navigation baseline."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from _common_args import add_max_steps, add_robot, add_seed


def main() -> int:
    parser = argparse.ArgumentParser()
    add_robot(parser, default="turtlebot4")
    parser.add_argument("--goal-x", type=float, default=1.0)
    parser.add_argument("--goal-y", type=float, default=-1.0)
    parser.add_argument("--goal-yaw", type=float, default=0.0)
    add_seed(parser)
    add_max_steps(parser, default=300)
    args = parser.parse_args()

    from physai.robots import navigate

    result = navigate(
        args.robot,
        goal_x=args.goal_x,
        goal_y=args.goal_y,
        goal_yaw=args.goal_yaw,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    print(
        f"robot={args.robot} reached={result.reached} steps={result.steps} "
        f"position_error={result.position_error:.4f} "
        f"heading_error={result.heading_error:.4f} "
        f"collisions={result.collision_count} "
        f"failure_reason={result.failure_reason or 'none'}"
    )
    return 0 if result.reached and result.collision_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
