"""Run the scripted planner on the current scene, then execute the plan.

    python scripts/plan_task.py --dry-run     # print the plan and stop
    python scripts/plan_task.py               # plan, then run it

`--dry-run` prints the plan without spending a rollout, so you can inspect
grounding quality on its own.
"""

from __future__ import annotations

import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
from _cli import new_parser
from _common_args import add_manifest, add_max_steps, add_seed, open_mujoco_session

from physai.planner import ScriptedPlanner
from physai.policy.plan_runner import PlanRunner


def main() -> int:
    ap = new_parser(__doc__)
    ap.add_argument(
        "--instruction",
        default="put the red cube on the green pad",
        help="natural-language task given to the planner",
    )
    add_manifest(ap, help="single-cube session manifest whose robot executes the plan")
    add_seed(ap)
    add_max_steps(ap, default=800)
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    ap.add_argument(
        "--out",
        type=Path,
        help="directory to save plan.json and, for debugging, the images the "
        "planner sees (frames/)",
    )
    args = ap.parse_args()

    session = open_mujoco_session(
        args.manifest, render=True, seed=args.seed, max_steps=args.max_steps
    )
    env = session.runtime.robot
    obs = env.reset(seed=args.seed)

    if args.out:
        import imageio.v3 as iio

        frames = args.out / "frames"
        frames.mkdir(parents=True, exist_ok=True)
        for name, frame in obs.images.items():
            iio.imwrite(frames / f"{name}.png", frame.data)
        print(f"frames -> {frames}")

    profile = env.grasp_profile()
    planner = ScriptedPlanner(env.cube_pos, env.target_pos, profile)
    plan = planner.plan(args.instruction, obs)
    print(f"\nplanner: {planner.name}")
    print(f"notes:   {plan.notes}\n")
    for i, sg in enumerate(plan.subgoals):
        xyz = sg.waypoint.pose.position.as_array()
        print(
            f"  {i}. {sg.skill:<12} {np.round(xyz, 3)}  grip={sg.gripper}"
            f"  <- {sg.target_description}"
        )
        if sg.rationale:
            print(f"     {sg.rationale}")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "plan.json").write_text(
            json.dumps(plan.to_dict(), indent=2), encoding="utf-8"
        )
        print(f"\nplan -> {args.out / 'plan.json'}")

    if args.dry_run or not plan.subgoals:
        session.close()
        return 0

    print(
        f"\nGround truth for comparison: cube={np.round(env.cube_pos, 3)} "
        f"target={np.round(env.target_pos, 3)}"
    )

    runner = PlanRunner(
        env.kin,
        plan,
        dt=env.control_dt,
        approach_dir=profile.approach_dir,
        ik_kwargs=profile.ik_kwargs,
        locate=lambda name: env.cube_pos if name == "cube" else None,
    )
    runner.reset(obs)
    info: dict = {}
    for _ in range(args.max_steps):
        obs, _, terminated, truncated, info = env.step(runner.act(obs))
        runner.note_progress(
            env.kin.pinch_center(env.data),
            env.joint_to_gripper(obs.joint_state.position[len(env.kin.joint_names)]),
        )
        if runner.done or terminated or truncated:
            break

    print(
        f"\nexecuted {runner.index}/{len(plan.subgoals)} sub-goals  "
        f"success={bool(info.get('success'))}  "
        f"dist_cube_target={info['dist_cube_target']:.3f}"
    )
    session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
