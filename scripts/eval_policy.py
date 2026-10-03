"""Evaluate any Policy over N seeds and report a success rate.

    python scripts/eval_policy.py --policy scripted --episodes 20
    python scripts/eval_policy.py --policy replay --dataset data/pickplace_v1

`replay` re-runs recorded actions through the sim. If replay succeeds but your
VLA does not, the problem is the model. If replay itself fails, the problem is
your action space, units, or control rate — check that before blaming training.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
from _common_args import (
    add_camera_resolution,
    add_checkpoint,
    add_episodes,
    add_max_steps,
    add_policy,
    add_robot,
    add_seed,
    add_simulator,
)

from physai.contracts import DEFAULT_CAMERA_RESOLUTION
from physai.control import SafetyViolation
from physai.data import EvaluationReport, load_episode
from physai.policy import available_policies, create_policy
from physai.robots import create_robot
from physai.robots.so101 import EnvConfig
from physai.sim.mujoco import SingleCubeFixedPlaceSceneConfig, SortingMinimalSceneConfig
from physai.sim.mujoco.domain_randomization import DomainRandomizationConfig
from physai.tasks import TaskRuntime, create_task

# Registers so101's "scripted"/"visual_servo" policies and the checkpoint-
# backed "lerobot" policy with their registries; --policy may select any of
# them, so all load eagerly (none import torch/lerobot at module scope).
import research.classical_control.so101_visual_servo  # noqa: E402,F401
import research.imitation_learning.vla_adapter  # noqa: E402,F401
import research.scripted_experts.so101_pick_place_expert  # noqa: E402,F401


def _parse_policy_arg(text: str) -> tuple[str, Any]:
    """Split `KEY=VALUE`, reading VALUE as a Python literal (a bare word stays a string)."""
    key, separator, raw = text.partition("=")
    if not separator or not key:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {text!r}")
    try:
        return key, ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return key, raw


def _format_distance(distance: float | None) -> str:
    return "n/a" if distance is None else f"{distance:.3f}"


def _parse_seed_list(text: str) -> list[int]:
    try:
        seeds = [int(part) for part in text.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected integers, got {text!r}") from None
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be distinct")
    return seeds


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
    ap.add_argument(
        "--camera-shift-unknown",
        action="store_true",
        help="with --camera-jitter, the policy keeps the nominal camera calibration "
        "while the camera has moved (default: it is told the shifted pose)",
    )
    ap.add_argument(
        "--lighting-scale",
        type=float,
        default=1.0,
        help="scale every MuJoCo light by this factor (difficulty sweep)",
    )
    ap.add_argument(
        "--clutter-count",
        type=int,
        default=0,
        help="place this many distractor boxes on the table, off the cube and target",
    )
    ap.add_argument(
        "--nominal-physics",
        action="store_true",
        help="with randomization on, leave friction and mass nominal so only the "
        "requested difficulty axis varies",
    )
    ap.add_argument("--dataset", type=Path, help="required for --policy replay")
    add_checkpoint(ap, help="required for --policy lerobot")
    add_camera_resolution(ap)
    add_simulator(
        ap,
        help="simulator engine (default mujoco); isaac runs the same scene, seeds "
        "and task on Isaac Sim (visual_servo only, no randomization)",
    )
    ap.add_argument(
        "--policy-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        type=_parse_policy_arg,
        help="override a policy constructor option, e.g. "
        "--policy-arg final_camera=front "
        "--policy-arg grasp_offset_xy='(0.0, 0.0)'; repeatable",
    )
    ap.add_argument(
        "--seeds",
        type=_parse_seed_list,
        help="comma-separated episode seeds (for example 5,13,28) instead of "
        "--seed .. --seed + --episodes",
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
    if args.seeds is not None:
        args.episodes = len(args.seeds)
    episode_seeds = args.seeds or [args.seed + ep for ep in range(args.episodes)]

    # An image-conditioned policy cannot run without rendered cameras: render
    # defaulted to --render alone (so `--policy lerobot` died on an empty
    # images dict) once env construction moved behind create_robot().
    needs_images = args.policy in {"lerobot", "visual_servo"}
    scene_type = (
        SortingMinimalSceneConfig if args.sorting else SingleCubeFixedPlaceSceneConfig
    )
    if args.camera_jitter < 0:
        ap.error("--camera-jitter must be non-negative")
    if args.lighting_scale <= 0:
        ap.error("--lighting-scale must be positive")
    if args.clutter_count < 0:
        ap.error("--clutter-count must be non-negative")
    difficulty = (
        args.camera_jitter > 0 or args.lighting_scale != 1.0 or args.clutter_count > 0
    )
    physics = {"friction_scale": (1.0, 1.0), "mass_scale": (1.0, 1.0)}
    randomization = DomainRandomizationConfig(
        enabled=difficulty,
        camera_position_jitter=args.camera_jitter,
        camera_shift_calibrated=not args.camera_shift_unknown,
        lighting_scale=(args.lighting_scale, args.lighting_scale),
        **(physics if args.nominal_physics else {}),
    )
    scene = scene_type(
        camera_resolution=args.camera_resolution or DEFAULT_CAMERA_RESOLUTION,
        clutter_count=args.clutter_count,
    )
    if args.simulator == "isaac":
        # The same scene config MuJoCo builds from, so both engines share the
        # table, cube, target, cameras and per-seed cube layout.
        if args.sorting or args.clutter_count:
            ap.error("--sim isaac supports neither --sorting nor --clutter-count")
        if args.camera_jitter > 0 and not args.camera_shift_unknown:
            ap.error(
                "--sim isaac moves cameras without telling the policy: "
                "add --camera-shift-unknown to --camera-jitter"
            )
        if args.policy not in {"visual_servo", "constant"}:
            ap.error("--sim isaac supports --policy visual_servo or constant")
        from physai.robots.so101.isaac_env import IsaacEnvConfig

        robot = create_robot(
            args.robot,
            simulator="isaac",
            config=IsaacEnvConfig(
                scene=scene,
                camera_resolution=scene.camera_resolution,
                lighting_scale=args.lighting_scale,
                camera_position_jitter=args.camera_jitter,
                cameras=("front", "wrist"),
                seed=args.seed,
                max_steps=args.max_steps,
                render=True,
            ),
        )
    else:
        robot = create_robot(
            args.robot,
            config=EnvConfig(
                scene=scene,
                seed=args.seed,
                max_steps=args.max_steps,
                render=args.render or needs_images,
                domain_randomization=randomization,
            ),
        )
    env = TaskRuntime(
        robot,
        create_task("sorting" if args.sorting else "single_cube_fixed_place"),
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
            overlap = sorted(train_seeds & set(episode_seeds))
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
        policy_kwargs = {"env": env, **dict(args.policy_arg)}
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
            seed = entry.get("seed", episode_seeds[ep])
            policy = create_policy("replay", env=env, actions=data["action"])
        else:
            seed = episode_seeds[ep]
            policy = reusable_policy

        obs = env.reset(seed=seed)
        # Isaac Sim sometimes stops drawing the robot (a startup glitch);
        # camera policies then fail for a reason unrelated to the policy, so
        # stop instead of recording those episodes as failures.
        rendered = getattr(robot, "robot_is_rendered", None)
        if rendered is not None and not rendered():
            raise SystemExit(
                f"episode {ep} (seed {seed}): Isaac Sim is not drawing the robot; "
                "restart the evaluation (results so far are valid)"
            )
        policy.reset(obs)
        total, info = 0.0, {}
        violation = None
        for _ in range(args.max_steps):
            try:
                obs, reward, terminated, truncated, info = env.step(policy.act(obs))
            except SafetyViolation as exc:
                # The gate refused the action: the episode ends as an unsafe
                # action rather than taking the whole evaluation down.
                violation = str(exc)
                break
            total += reward
            if terminated or truncated:
                break

        results.append(
            {
                "seed": seed,
                "success": violation is None and bool(info.get("success")),
                "steps": env.step_count,
                "reward": total,
                "return": total,
                "timeout": violation is None and bool(truncated or info.get("timeout")),
                "collision": bool(
                    info.get("collision") or info.get("collision_detected")
                ),
                "unsafe_action": violation is not None
                or bool(info.get("unsafe_action")),
                "held_out": bool(train_seeds) and seed not in train_seeds,
                "dist_cube_target": info.get("dist_cube_target"),
                **(
                    {
                        "visual_error_px": policy.metrics.visual_error_px,
                        "ee_error_m": policy.metrics.ee_error_m,
                        "phase": policy.metrics.phase,
                        "settling_time_s": policy.metrics.settling_time_s,
                        "grasp_retries": policy.metrics.grasp_retries,
                        "failure_reason": violation or policy.metrics.failure_reason,
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
            f"d={_format_distance(results[-1]['dist_cube_target'])}"
            + (f"  UNSAFE: {violation}" if violation else "")
        )

    env.close()
    report = EvaluationReport(
        policy=args.policy,
        robot=args.robot,
        task="sorting" if args.sorting else "single_cube_fixed_place",
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
        payload["seed_start"] = episode_seeds[0]
        payload["policy_args"] = {key: repr(value) for key, value in args.policy_arg}
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"json -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
