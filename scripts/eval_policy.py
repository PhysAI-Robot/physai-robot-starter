"""Measure a Policy: run it over N seeds of one session and report a success rate.

    python scripts/eval_policy.py --episodes 20              # the robot's demo policy
    python scripts/eval_policy.py --policy replay --dataset data/pickplace_v1

For looking at a run (viewer, web host, one quick episode) use `run_sim.py`;
this script is for numbers you compare. It runs the session in a manifest
(default: configs/manifests/so101_single_cube_fixed_place.yaml, or
so101_sorting.yaml with `--sorting`) and its flags only turn the difficulty
knobs, so MuJoCo and Isaac Sim evaluate the same scene, seeds and task.

`replay` re-runs recorded actions through the sim. If replay succeeds but your
VLA does not, the problem is the model. If replay itself fails, the problem is
your action space, units, or control rate — check that before blaming training.

A policy from your own module is named with `--policy` (or in the manifest)
and needs that module in the manifest's `plugins:` list.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np
from _cli import new_parser
from _common_args import (
    DEFAULT_MANIFEST,
    SORTING_MANIFEST,
    add_camera_resolution,
    add_episodes,
    add_max_steps,
    add_policy,
    add_policy_args,
    add_run_outputs,
    add_seed,
    add_simulator,
    add_list_policies,
    policy_kwargs,
    print_policies,
)

from _outputs import RunOutputs, wants_cameras
from _recording import dataset_metadata
from physai.config import DomainRandomizationConfig, load_manifest
from physai.config.compat import with_overrides
from physai.data import EvaluationReport, load_episode
from physai.data.evaluation import trajectory_metrics
from physai.policy import create_policy, policy_traits
from physai.runtime import (
    EpisodeObserver,
    RenderGlitch,
    create_session,
    run_episode,
)


def _format_distance(distance: float | None) -> str:
    return "n/a" if distance is None else f"{distance:.3f}"


def _seed_overlap_warning(
    train_seeds: set[int], episode_seeds: list[int]
) -> str | None:
    """A warning when the evaluation reuses seeds the policy was trained on, else None."""
    overlap = sorted(train_seeds & set(episode_seeds))
    if not overlap:
        return None
    shown = f"{overlap[:8]}{'...' if len(overlap) > 8 else ''}"
    return (
        f"WARNING: {len(overlap)}/{len(episode_seeds)} evaluation seeds were in "
        f"this policy's training set {shown}\n"
        f"         This measures memorisation, not generalisation. "
        f"Pick a --seed beyond {max(train_seeds)}."
    )


def _policy_metrics(policy: Any, violation: str | None) -> dict[str, Any]:
    """The policy's own per-episode fields; a safety violation is its failure reason."""
    metrics = dict(policy.episode_metrics())
    if violation and "failure_reason" in metrics:
        metrics["failure_reason"] = violation
    return metrics


def _parse_seed_list(text: str) -> list[int]:
    try:
        seeds = [int(part) for part in text.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected integers, got {text!r}") from None
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be distinct")
    return seeds


class _EpisodeTrace(EpisodeObserver):
    """Where the episode started (single-cube scenes) and how the gripper moved."""

    def __init__(self, robot: Any) -> None:
        self._robot = robot
        self.pose: dict[str, list[float]] = {}
        self._ee: list[tuple[float, float, float]] = []

    def metrics(self) -> dict[str, float]:
        if len(self._ee) < 2:
            return {}
        return trajectory_metrics(
            np.array(self._ee), 1.0 / getattr(self._robot.cfg, "control_hz", 30)
        )

    def _track(self, observation) -> None:
        if observation.ee_pose is not None:
            p = observation.ee_pose.pose.position
            self._ee.append((p.x, p.y, p.z))

    def after_step(self, observation, action, next_observation, *_) -> None:
        self._track(next_observation)

    def on_reset(self, observation) -> None:
        self._track(observation)
        if getattr(self._robot, "cube_positions", None):
            return
        for key, attr in (("cube_start", "cube_pos"), ("target_pos", "target_pos")):
            value = getattr(self._robot, attr, None)
            if value is not None:
                self.pose[key] = [round(float(v), 4) for v in value[:2]]


def main() -> int:
    ap = new_parser(__doc__)
    ap.add_argument(
        "--manifest",
        type=Path,
        help=f"session manifest to evaluate (default: {DEFAULT_MANIFEST}, or "
        f"{SORTING_MANIFEST} with --sorting)",
    )
    # No `choices`: a plugin the manifest loads can add policies, so the name
    # is checked after the manifest is loaded (physai.config.compat).
    add_policy(ap, default="demo")
    add_episodes(ap, default=20)
    add_seed(ap)
    add_max_steps(ap, help="override the episode length (default: the manifest's)")
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
    ap.add_argument(
        "--save-dataset",
        type=Path,
        metavar="DIR",
        help="also write every episode as a training dataset (episode_NNNNN.npz + "
        "meta.json) into DIR, e.g. replayed demonstrations seen through another simulator",
    )
    add_policy_args(ap)
    add_camera_resolution(ap)
    add_simulator(
        ap,
        help="simulator engine (default: the manifest's, mujoco); isaac runs the same "
        "scene, seeds and task on Isaac Sim (policies that support it; "
        "--list-policies shows which; single-cube scenes only)",
    )
    add_list_policies(ap)
    ap.add_argument(
        "--seeds",
        type=_parse_seed_list,
        help="comma-separated episode seeds (for example 5,13,28) instead of "
        "--seed .. --seed + --episodes",
    )
    add_run_outputs(ap)
    ap.add_argument(
        "--sorting",
        action="store_true",
        help="evaluate the three-cube sorting task instead of pick-and-place, "
        "matching the sorting data-collection scripts",
    )
    ap.add_argument(
        "--json", type=Path, help="write the full per-episode results to this JSON file"
    )
    args = ap.parse_args()
    if args.list_policies:
        return print_policies(
            args.manifest or (SORTING_MANIFEST if args.sorting else DEFAULT_MANIFEST)
        )
    if args.seeds is not None:
        args.episodes = len(args.seeds)
    episode_seeds = args.seeds or [args.seed + ep for ep in range(args.episodes)]

    if args.manifest and args.sorting:
        ap.error(
            "--sorting selects its own manifest; it cannot be used with --manifest"
        )
    if args.camera_jitter < 0:
        ap.error("--camera-jitter must be non-negative")
    if args.lighting_scale <= 0:
        ap.error("--lighting-scale must be positive")
    if args.clutter_count < 0:
        ap.error("--clutter-count must be non-negative")
    if args.policy == "replay":
        if not args.dataset:
            ap.error("--policy replay needs --dataset")
        meta = json.loads((args.dataset / "meta.json").read_text(encoding="utf-8"))
        replay_episodes = meta["episodes"]
        if not replay_episodes:
            ap.error(f"no episodes in {args.dataset}")

    manifest = load_manifest(
        args.manifest or (SORTING_MANIFEST if args.sorting else DEFAULT_MANIFEST)
    )
    simulator = args.simulator or manifest.simulator
    difficulty = (
        args.camera_jitter > 0 or args.lighting_scale != 1.0 or args.clutter_count > 0
    )
    overrides: dict[str, Any] = {}
    if simulator == "isaac":
        # The same scene config MuJoCo builds from, so both engines share the
        # table, cube, target, cameras and per-seed cube layout.
        if args.sorting or args.clutter_count:
            ap.error("--sim isaac supports neither --sorting nor --clutter-count")
        if args.camera_jitter > 0 and not args.camera_shift_unknown:
            ap.error(
                "--sim isaac moves cameras without telling the policy: "
                "add --camera-shift-unknown to --camera-jitter"
            )
        if difficulty:
            overrides["robot_config"] = {
                "lighting_scale": args.lighting_scale,
                "camera_position_jitter": args.camera_jitter,
            }
    elif difficulty:
        physics = {"friction_scale": (1.0, 1.0), "mass_scale": (1.0, 1.0)}
        overrides["domain_randomization"] = DomainRandomizationConfig(
            enabled=True,
            camera_position_jitter=args.camera_jitter,
            camera_shift_calibrated=not args.camera_shift_unknown,
            lighting_scale=(args.lighting_scale, args.lighting_scale),
            **(physics if args.nominal_physics else {}),
        )
        if args.clutter_count:
            overrides["scene_overrides"] = {"clutter_count": args.clutter_count}
    try:
        manifest = with_overrides(
            manifest,
            seed=args.seed,
            max_steps=args.max_steps,
            camera_resolution=args.camera_resolution,
            # replay builds its policy per episode, from that episode's actions
            policy="idle" if args.policy == "replay" else args.policy,
            simulator=args.simulator,
            **overrides,
        )
    except ValueError as exc:
        ap.error(str(exc))
    robot_name = manifest.robots[0].robot
    task_name = manifest.task_for(manifest.robots[0])

    # An image-conditioned policy cannot run without rendered cameras.
    needs_images = policy_traits(args.policy, robot_name).needs_images
    # Built once outside the loop where possible — reloading the checkpoint
    # from disk per episode would dominate wall-clock time for no reason.
    try:
        session = create_session(
            manifest,
            render=simulator == "isaac"
            or needs_images
            or wants_cameras(args)
            or args.save_dataset is not None,
            policy_kwargs=policy_kwargs(args),
        )
    except ValueError as exc:
        ap.error(str(exc))
    runtime = session.runtime
    env = runtime.robot
    fps = getattr(env.cfg, "control_hz", 30)
    outputs = RunOutputs(
        runtime,
        args,
        simulator=simulator,
        robot=robot_name,
        policy=args.policy,
        task=task_name,
        fps=fps,
        dataset_dir=args.save_dataset,
        metadata=(
            dataset_metadata(runtime, manifest)
            if args.save_dataset is not None
            else None
        ),
    )

    # Evaluating on seeds the policy trained on measures recall, not
    # generalisation, and the two can differ by a lot: a checkpoint can do well
    # on seeds it trained on and fail on held-out ones. Checkpoints written
    # before train_seeds was recorded simply skip this check.
    train_seeds = set(getattr(runtime.policy, "training_seeds", None) or ())
    warning = _seed_overlap_warning(train_seeds, episode_seeds)
    if warning:
        print(warning)

    results = []
    for ep in range(args.episodes):
        if args.policy == "replay":
            entry = replay_episodes[(args.seed + ep) % len(replay_episodes)]
            data = load_episode(args.dataset / entry["file"])
            seed = entry.get("seed", episode_seeds[ep])
            runtime.policy = create_policy("replay", env=env, actions=data["action"])
        else:
            seed = episode_seeds[ep]

        trace = _EpisodeTrace(runtime.robot)
        try:
            outcome = run_episode(runtime, seed, (*outputs.begin(), trace))
        except RenderGlitch as exc:
            # Camera policies would fail for a reason unrelated to the policy,
            # so stop instead of recording those episodes as failures.
            raise SystemExit(
                f"episode {ep} ({exc}); restart the evaluation "
                "(results so far are valid)"
            ) from exc
        info, violation, policy = outcome.info, outcome.violation, runtime.policy

        results.append(
            {
                "seed": seed,
                "success": outcome.success,
                "steps": outcome.steps,
                "reward": outcome.reward,
                "return": outcome.reward,
                "timeout": violation is None
                and bool(outcome.truncated or info.get("timeout")),
                "collision": bool(
                    info.get("collision") or info.get("collision_detected")
                ),
                "unsafe_action": violation is not None
                or bool(info.get("unsafe_action")),
                "held_out": bool(train_seeds) and seed not in train_seeds,
                "dist_cube_target": info.get("dist_cube_target"),
                # Where the episode started, for failures by workspace region.
                **trace.pose,
                **trace.metrics(),
                **_policy_metrics(policy, violation),
                # Which cube the episode asked for, so a per-color breakdown is
                # possible after the fact. ACT never receives this.
                **(
                    {"target_color": info["target_color"]}
                    if task_name == "sorting"
                    else {}
                ),
            }
        )
        print(
            f"ep {ep:3d} seed={seed:<5d} success={results[-1]['success']!s:<5} "
            f"steps={outcome.steps:<4d} return={outcome.reward:7.2f} "
            f"d={_format_distance(results[-1]['dist_cube_target'])}"
            + (f"  UNSAFE: {violation}" if violation else "")
        )
        outputs.save(results[-1]["success"], seed)

    outputs.close()
    session.close()
    report = EvaluationReport(
        policy=args.policy,
        robot=robot_name,
        task=task_name,
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

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = report.to_dict()
        payload["checkpoint"] = str(args.checkpoint) if args.checkpoint else None
        payload["seed_start"] = episode_seeds[0]
        payload["policy_args"] = {key: repr(value) for key, value in args.policy_arg}
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"json -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
