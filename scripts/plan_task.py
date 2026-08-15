"""Run the VLM planner on the current scene, then execute the plan.

    # offline, no API key needed — checks the planner->policy plumbing
    python scripts/plan_task.py --planner scripted

    # real VLM
    set ANTHROPIC_API_KEY=...          # PowerShell: $env:ANTHROPIC_API_KEY="..."
    python scripts/plan_task.py --instruction "put the red cube on the green pad"

`--dry-run` prints the plan and stops, so you can inspect grounding quality
without spending a rollout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from physai.planner import ScriptedPlanner
from physai.policy.plan_runner import PlanRunner
from physai.sim import EnvConfig, SO101PickPlaceEnv, SceneConfig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruction", default="put the red cube on the green pad")
    ap.add_argument("--planner", default="claude", choices=["claude", "scripted"])
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--effort", default="medium",
                    choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=800)
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    ap.add_argument("--save-plan", type=Path)
    ap.add_argument("--save-frames", type=Path,
                    help="write the images sent to the planner, for debugging")
    args = ap.parse_args()

    env = SO101PickPlaceEnv(EnvConfig(
        scene=SceneConfig(camera_width=512, camera_height=384),
        seed=args.seed, max_steps=args.max_steps, render=True,
    ))
    obs = env.reset(seed=args.seed)

    if args.save_frames:
        import imageio.v3 as iio

        args.save_frames.mkdir(parents=True, exist_ok=True)
        for name, frame in obs.images.items():
            iio.imwrite(args.save_frames / f"{name}.png", frame.data)
        print(f"frames -> {args.save_frames}")

    if args.planner == "claude":
        from physai.planner import ClaudePlanner

        planner = ClaudePlanner(model=args.model, effort=args.effort)
    else:
        planner = ScriptedPlanner(env.cube_pos, env.target_pos)

    plan = planner.plan(args.instruction, obs)
    print(f"\nplanner: {planner.name}")
    print(f"notes:   {plan.notes}\n")
    for i, sg in enumerate(plan.subgoals):
        xyz = sg.waypoint.pose.position.as_array()
        print(f"  {i}. {sg.skill:<12} {np.round(xyz, 3)}  grip={sg.gripper}"
              f"  <- {sg.target_description}")
        if sg.rationale:
            print(f"     {sg.rationale}")

    if args.save_plan:
        args.save_plan.parent.mkdir(parents=True, exist_ok=True)
        args.save_plan.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
        print(f"\nplan -> {args.save_plan}")

    if args.dry_run or not plan.subgoals:
        env.close()
        return 0

    print(f"\nGround truth for comparison: cube={np.round(env.cube_pos, 3)} "
          f"target={np.round(env.target_pos, 3)}")

    runner = PlanRunner(env.kin, plan, dt=env.control_dt)
    runner.reset(obs)
    info: dict = {}
    for _ in range(args.max_steps):
        obs, _, terminated, truncated, info = env.step(runner.act(obs))
        runner.note_progress(
            env.kin.pinch_center(env.data),
            env.joint_to_gripper(obs.joint_state.position[5]),
        )
        if runner.done or terminated or truncated:
            break

    print(f"\nexecuted {runner.index}/{len(plan.subgoals)} sub-goals  "
          f"success={bool(info.get('success'))}  "
          f"dist_cube_target={info['dist_cube_target']:.3f}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
