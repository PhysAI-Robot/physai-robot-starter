"""Compare deterministic and randomized scripted SO-101 evaluation.

uv run python scripts/eval_randomization.py --episodes 20 --clutter-count 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from _cli import new_parser
from _common_args import DEFAULT_MANIFEST, add_episodes, add_max_steps, add_seed

from physai.config import DomainRandomizationConfig, load_manifest
from physai.config.compat import with_overrides
from physai.runtime import create_session, run_episode

# Registers so101's "scripted" policy with the policy registry.
import research.scripted_experts.so101_pick_place_expert  # noqa: F401


def evaluate_mode(args: argparse.Namespace, randomized: bool) -> dict:
    config = DomainRandomizationConfig(
        enabled=randomized,
        clutter_x_range=(0.14, 0.28),
        clutter_y_range=(-0.16, 0.16),
        camera_position_jitter=0.005 if randomized else 0.0,
    )
    manifest = with_overrides(
        load_manifest(DEFAULT_MANIFEST),
        seed=args.seed,
        max_steps=args.max_steps,
        policy="scripted",
        domain_randomization=config,
        scene_overrides={"clutter_count": args.clutter_count}
        if args.clutter_count
        else None,
    )
    session = create_session(manifest, render=False)
    runtime = session.runtime
    results = []
    try:
        for episode in range(args.episodes):
            seed = args.seed + episode
            outcome = run_episode(runtime, seed)
            results.append(
                {
                    "seed": seed,
                    "success": outcome.success,
                    "steps": outcome.steps,
                    "return": outcome.reward,
                    "randomization": runtime.robot.randomization_metadata.as_dict(),
                }
            )
    finally:
        session.close()
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
    parser = new_parser(__doc__)
    add_episodes(parser, default=20)
    add_seed(parser)
    add_max_steps(parser, help="override the episode length (default: the manifest's)")
    parser.add_argument(
        "--clutter-count",
        type=int,
        default=0,
        help="distractor boxes placed on the table, off the cube and target",
    )
    parser.add_argument(
        "--json-out", type=Path, help="write both modes' results as JSON to this file"
    )
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
