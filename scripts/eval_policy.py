"""Evaluate any Policy over N seeds and report a success rate.

    python scripts/eval_policy.py --policy scripted --episodes 20
    python scripts/eval_policy.py --policy replay --dataset data/pickplace_v1

`replay` re-runs recorded actions through the sim. If replay succeeds but your
VLA does not, the problem is the model. If replay itself fails, the problem is
your action space, units, or control rate — check that before blaming training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from physai.data import load_episode
from physai.policy import ConstantPolicy, ReplayPolicy, ScriptedPickPlace
from physai.robots import available_robots, create_robot
from physai.sim import EnvConfig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="so101", choices=["so101"],
                    help="eval_policy currently supports the SO-101 manipulation workflow")
    ap.add_argument("--policy", default="scripted",
                    choices=["scripted", "constant", "replay", "lerobot"])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--dataset", type=Path, help="required for --policy replay")
    ap.add_argument("--checkpoint", type=Path, help="required for --policy lerobot")
    ap.add_argument("--camera-size", type=int, default=224,
                    help="env camera render resolution (downscaled to the "
                         "checkpoint's training size internally)")
    ap.add_argument("--render", action="store_true",
                    help="render cameras (slower; needed for image-conditioned policies)")
    ap.add_argument("--json-out", type=Path, help="write full per-episode results as JSON")
    args = ap.parse_args()

    env = create_robot(args.robot, config=EnvConfig(seed=args.seed, max_steps=args.max_steps,
                                                    render=args.render))

    episodes = None
    if args.policy == "replay":
        if not args.dataset:
            ap.error("--policy replay needs --dataset")
        import json

        meta = json.loads((args.dataset / "meta.json").read_text(encoding="utf-8"))
        episodes = meta["episodes"]
        if not episodes:
            ap.error(f"no episodes in {args.dataset}")

    if args.policy == "lerobot" and not args.checkpoint:
        ap.error("--policy lerobot needs --checkpoint")

    # Built once outside the loop where possible — reloading the checkpoint
    # from disk per episode would dominate wall-clock time for no reason.
    reusable_policy = None
    if args.policy == "scripted":
        reusable_policy = ScriptedPickPlace(env.kin, env)
    elif args.policy == "constant":
        reusable_policy = ConstantPolicy()
    elif args.policy == "lerobot":
        reusable_policy = LeRobotPolicy.from_checkpoint(env, args.checkpoint)

    results = []
    for ep in range(args.episodes):
        if args.policy == "replay":
            entry = episodes[ep % len(episodes)]
            data = load_episode(args.dataset / entry["file"])
            seed = entry.get("seed", args.seed + ep)
            policy = ReplayPolicy(env, data["action"])
        else:
            seed = args.seed + ep
            policy = reusable_policy

        obs = env.reset(seed=seed)
        policy.reset(obs)
        total, info = 0.0, {}
        for _ in range(args.max_steps):
            obs, reward, terminated, truncated, info = env.step(policy.act(obs))
            total += reward
            if terminated or truncated:
                break

        results.append({
            "seed": seed,
            "success": bool(info.get("success")),
            "steps": env.step_count,
            "return": total,
            "dist_cube_target": info["dist_cube_target"],
        })
        print(f"ep {ep:3d} seed={seed:<5d} success={results[-1]['success']!s:<5} "
              f"steps={env.step_count:<4d} return={total:7.2f} "
              f"d={results[-1]['dist_cube_target']:.3f}")

    env.close()
    ok = sum(r["success"] for r in results)
    n = len(results)
    print(f"\npolicy={args.policy}  success {ok}/{n} = {ok / n:.0%}")
    print(f"mean return {np.mean([r['return'] for r in results]):.2f}   "
          f"mean steps {np.mean([r['steps'] for r in results]):.0f}")

    if args.json_out:
        import json

        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({
            "policy": args.policy,
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "episodes": args.episodes,
            "seed": args.seed,
            "success_count": ok,
            "success_rate": ok / n,
            "mean_return": float(np.mean([r["return"] for r in results])),
            "mean_steps": float(np.mean([r["steps"] for r in results])),
            "results": results,
        }, indent=2), encoding="utf-8")
        print(f"json -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
