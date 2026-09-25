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
from _common_args import (
    add_checkpoint,
    add_episodes,
    add_max_steps,
    add_policy,
    add_robot,
    add_seed,
)

from physai.data import EvaluationReport, load_episode
from physai.policy import available_policies, create_policy
from physai.robots import create_robot
from physai.robots.so101 import EnvConfig
from physai.sim import PickPlaceMinimalSceneConfig, SortingMinimalSceneConfig
from physai.sim.domain_randomization import DomainRandomizationConfig
from physai.tasks import TaskRuntime, create_task

# Registers so101's "scripted"/"visual_servo" policies and the checkpoint-
# backed "lerobot" policy with their registries; --policy may select any of
# them, so all load eagerly (none import torch/lerobot at module scope).
import research.classical_control.so101_visual_servo  # noqa: E402,F401
import research.imitation_learning.vla_adapter  # noqa: E402,F401
import research.scripted_experts.so101_pick_place_expert  # noqa: E402,F401


def main() -> int:
    ap = argparse.ArgumentParser()
    add_robot(
        ap,
        default="so101",
        choices=["so101"],
        help="eval_policy currently supports the SO-101 manipulation workflow",
    )
    add_policy(ap, default="scripted", choices=available_policies())
    add_episodes(ap, default=20)
    add_seed(ap)
    add_max_steps(ap, default=600)
    ap.add_argument(
        "--camera-jitter",
        type=float,
        default=0.0,
        help="enable seeded camera-position jitter in metres for robustness evaluation",
    )
    ap.add_argument("--dataset", type=Path, help="required for --policy replay")
    add_checkpoint(ap, help="required for --policy lerobot")
    ap.add_argument(
        "--camera-size",
        type=int,
        default=224,
        help="env camera render resolution (downscaled to the "
        "checkpoint's training size internally)",
    )
    ap.add_argument(
        "--render",
        action="store_true",
        help="render cameras (slower; needed for image-conditioned policies)",
    )
    ap.add_argument(
        "--sorting",
        action="store_true",
        help="evaluate the three-cube sorting task instead of pick-and-place, "
        "matching `collect_demos.py --sorting`",
    )
    ap.add_argument(
        "--json-out", type=Path, help="write full per-episode results as JSON"
    )
    args = ap.parse_args()

    # An image-conditioned policy cannot run without rendered cameras, and it
    # must see them at the resolution --camera-size asks for. Both were lost
    # when env construction moved behind create_robot(): render defaulted to
    # --render alone (so `--policy lerobot` died on an empty images dict) and
    # --camera-size stopped reaching the scene config entirely.
    needs_images = args.policy in {"lerobot", "visual_servo"}
    scene_kwargs = {"camera_width": args.camera_size, "camera_height": args.camera_size}
    scene_type = (
        SortingMinimalSceneConfig if args.sorting else PickPlaceMinimalSceneConfig
    )
    if args.camera_jitter < 0:
        ap.error("--camera-jitter must be non-negative")
    randomization = DomainRandomizationConfig(
        enabled=args.camera_jitter > 0,
        camera_position_jitter=args.camera_jitter,
    )
    robot = create_robot(
        args.robot,
        config=EnvConfig(
            scene=scene_type(**scene_kwargs),
            seed=args.seed,
            max_steps=args.max_steps,
            render=args.render or needs_images,
            domain_randomization=randomization,
        ),
    )
    env = TaskRuntime(
        robot,
        create_task("sorting" if args.sorting else "pick_place"),
    )

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

    # Evaluating on seeds the policy trained on measures recall, not
    # generalisation, and the two can differ by a lot: a sorting checkpoint
    # scored 70% on seeds that were 80% training layouts and 10% on held-out
    # ones. Checkpoints written before train_seeds was recorded simply skip
    # this check.
    if args.policy == "lerobot":
        import json

        meta_path = args.checkpoint / "training_meta.json"
        if meta_path.exists():
            train_seeds = set(
                json.loads(meta_path.read_text(encoding="utf-8")).get("train_seeds")
                or []
            )
            overlap = sorted(
                train_seeds & set(range(args.seed, args.seed + args.episodes))
            )
            if overlap:
                print(
                    f"WARNING: {len(overlap)}/{args.episodes} evaluation seeds were in "
                    f"this checkpoint's training set {overlap[:8]}"
                    f"{'...' if len(overlap) > 8 else ''}\n"
                    f"         This measures memorisation, not generalisation. "
                    f"Pick a --seed beyond {max(train_seeds)}."
                )

    # Built once outside the loop where possible — reloading the checkpoint
    # from disk per episode would dominate wall-clock time for no reason.
    reusable_policy = None
    if args.policy != "replay":
        policy_kwargs = {"env": env}
        if args.policy == "lerobot":
            policy_kwargs["checkpoint"] = args.checkpoint
        reusable_policy = create_policy(
            args.policy,
            **policy_kwargs,
        )

    results = []
    train_seeds: set[int] = set()
    if args.policy == "lerobot" and args.checkpoint:
        import json

        meta_path = args.checkpoint / "training_meta.json"
        if meta_path.exists():
            train_seeds = set(
                json.loads(meta_path.read_text(encoding="utf-8")).get("train_seeds")
                or []
            )
    for ep in range(args.episodes):
        if args.policy == "replay":
            entry = episodes[ep % len(episodes)]
            data = load_episode(args.dataset / entry["file"])
            seed = entry.get("seed", args.seed + ep)
            policy = create_policy("replay", env=env, actions=data["action"])
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

        results.append(
            {
                "seed": seed,
                "success": bool(info.get("success")),
                "steps": env.step_count,
                "reward": total,
                "return": total,
                "timeout": bool(truncated or info.get("timeout")),
                "collision": bool(
                    info.get("collision") or info.get("collision_detected")
                ),
                "unsafe_action": bool(info.get("unsafe_action")),
                "held_out": bool(train_seeds) and seed not in train_seeds,
                "dist_cube_target": info["dist_cube_target"],
                **(
                    {
                        "visual_error_px": policy.metrics.visual_error_px,
                        "ee_error_m": policy.metrics.ee_error_m,
                        "phase": policy.metrics.phase,
                        "settling_time_s": policy.metrics.settling_time_s,
                        "failure_reason": policy.metrics.failure_reason,
                    }
                    if args.policy == "visual_servo"
                    else {}
                ),
                # Which cube the episode asked for, so a per-color breakdown is
                # possible after the fact. ACT never receives this.
                **({"target_color": info["target_color"]} if args.sorting else {}),
            }
        )
        print(
            f"ep {ep:3d} seed={seed:<5d} success={results[-1]['success']!s:<5} "
            f"steps={env.step_count:<4d} return={total:7.2f} "
            f"d={results[-1]['dist_cube_target']:.3f}"
        )

    env.close()
    report = EvaluationReport(
        policy=args.policy,
        robot=args.robot,
        task="sorting" if args.sorting else "pick_place",
        results=tuple(results),
    )
    summary = report.summary
    print(
        f"\npolicy={args.policy}  success {summary['success_count']}/{summary['episodes']} = "
        f"{summary['success_rate']:.0%}"
    )
    print(
        f"mean reward {summary['mean_reward']:.2f}   mean steps {summary['mean_steps']:.0f}   "
        f"collisions {summary['collision_count']}   timeouts {summary['timeout_count']}   "
        f"unsafe actions {summary['unsafe_action_count']}"
    )
    if summary["held_out_success_rate"] is not None:
        print(
            f"held-out success {summary['held_out_success_rate']:.0%} "
            f"({summary['held_out_episodes']} episodes)"
        )

    if args.json_out:
        import json

        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = report.to_dict()
        payload["checkpoint"] = str(args.checkpoint) if args.checkpoint else None
        payload["seed_start"] = args.seed
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"json -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
