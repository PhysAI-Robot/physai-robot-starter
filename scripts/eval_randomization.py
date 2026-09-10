"""Compare deterministic and randomized scripted SO-101 evaluation.

    uv run python scripts/eval_randomization.py --episodes 20 --clutter-count 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from physai.config import DomainRandomizationConfig
from physai.policy import create_policy
from physai.robots import create_robot
from physai.robots.so101 import EnvConfig
from physai.sim import SceneConfig
from physai.tasks import TaskRuntime, create_task


def evaluate_mode(args: argparse.Namespace, randomized: bool) -> dict:
    config = DomainRandomizationConfig(
        enabled=randomized,
        clutter_x_range=(0.14, 0.28),
        clutter_y_range=(-0.16, 0.16),
        camera_position_jitter=0.005 if randomized else 0.0,
    )
    robot = create_robot(
        "so101",
        config=EnvConfig(
            scene=SceneConfig(
                camera_width=128,
                camera_height=128,
                clutter_count=args.clutter_count,
            ),
            max_steps=args.max_steps,
            render=False,
            domain_randomization=config,
        ),
    )
    env = TaskRuntime(robot, create_task("pick_place"))
    policy = create_policy("scripted", env=env)
    results = []
    try:
        for episode in range(args.episodes):
            seed = args.seed + episode
            observation = env.reset(seed=seed)
            policy.reset(observation)
            total = 0.0
            info: dict = {}
            for _ in range(args.max_steps):
                observation, reward, terminated, truncated, info = env.step(
                    policy.act(observation)
                )
                total += reward
                if terminated or truncated or policy.done:
                    break
            results.append({
                "seed": seed,
                "success": bool(info.get("success")),
                "steps": env.step_count,
                "return": total,
                "randomization": env.randomization_metadata.as_dict(),
            })
    finally:
        env.close()
    success_count = sum(item["success"] for item in results)
    return {
        "enabled": randomized,
        "episodes": len(results),
        "success_count": success_count,
        "success_rate": success_count / len(results),
        "mean_return": sum(item["return"] for item in results) / len(results),
        "mean_steps": sum(item["steps"] for item in results) / len(results),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--clutter-count", type=int, default=0)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.clutter_count < 0:
        parser.error("--clutter-count must be non-negative")

    report = {
        "episodes": args.episodes,
        "seed": args.seed,
        "clutter_count": args.clutter_count,
        "deterministic": evaluate_mode(args, randomized=False),
        "randomized": evaluate_mode(args, randomized=True),
    }
    for name in ("deterministic", "randomized"):
        mode = report[name]
        print(
            f"{name}: success {mode['success_count']}/{mode['episodes']} "
            f"= {mode['success_rate']:.0%}, "
            f"mean_return={mode['mean_return']:.2f}, "
            f"mean_steps={mode['mean_steps']:.0f}"
        )
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"json -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
