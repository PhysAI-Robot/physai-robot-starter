"""Generate demonstrations with the scripted expert.

This is the dataset you fine-tune a VLA on. Failed episodes are discarded by
default — behaviour cloning on failures teaches failure.

    python scripts/collect_demos.py --episodes 50 --dataset data/pickplace_v1
    python scripts/collect_demos.py --episodes 50 --keep-failures   # for analysis
    python scripts/collect_demos.py --manifest configs/manifests/so101_randomized_pick_place.yaml --episodes 200

Runs the same session as `eval_policy.py` (configs/manifests/so101_*.yaml), so a
policy trained on these demos is evaluated on the scene it learned.
"""

from __future__ import annotations

from pathlib import Path

import _bootstrap  # noqa: F401
from _cli import new_parser
from _common_args import (
    DEFAULT_MANIFEST,
    SORTING_MANIFEST,
    add_camera_resolution,
    add_episodes,
    add_max_steps,
    add_seed,
)
from _recording import RunRecorder, dataset_metadata

from physai.config import load_manifest
from physai.config.compat import with_overrides
from physai.contracts import parse_camera_resolution
from physai.runtime import create_session, run_episode
from research.scripted_experts.so101_pick_place_expert import SO101PickPlaceExpert


def main() -> int:
    ap = new_parser(__doc__)
    add_episodes(ap, default=20)
    ap.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/pickplace_v1"),
        help="dataset folder to write (episode_NNNNN.npz + meta.json)",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        help=f"session manifest to collect from (default: {DEFAULT_MANIFEST}, or "
        f"{SORTING_MANIFEST} with --sorting)",
    )
    add_seed(ap)
    add_max_steps(ap, help="override the episode length (default: the manifest's)")
    add_camera_resolution(ap)
    ap.add_argument(
        "--keep-failures",
        action="store_true",
        help="also keep episodes where the expert failed (default: discard them)",
    )
    ap.add_argument(
        "--no-images",
        action="store_true",
        help="record state/action only (much smaller files)",
    )
    ap.add_argument(
        "--task",
        default="put the red cube on the green pad",
        help="instruction stored with each episode",
    )
    ap.add_argument(
        "--sorting",
        action="store_true",
        help="3-cube color sorting variant. --task becomes a "
        "per-episode instruction naming the randomly chosen "
        "target color, e.g. 'put the blue cube on the green pad'.",
    )
    args = ap.parse_args()
    if args.manifest and args.sorting:
        ap.error(
            "--sorting selects its own manifest; it cannot be used with --manifest"
        )

    manifest = with_overrides(
        load_manifest(
            args.manifest or (SORTING_MANIFEST if args.sorting else DEFAULT_MANIFEST)
        ),
        seed=args.seed,
        max_steps=args.max_steps,
        camera_resolution=args.camera_resolution,
    )
    session = create_session(manifest, render=not args.no_images)
    runtime = session.runtime
    env = runtime.robot
    width, height = parse_camera_resolution(manifest.simulation.camera_resolution)
    recorder = RunRecorder(
        runtime,
        fps=env.cfg.control_hz,
        name="demos",
        task=args.task,
        dataset_dir=args.dataset,
        metadata=dataset_metadata(runtime, manifest, store_images=not args.no_images),
    )

    attempted = kept = 0
    while kept < args.episodes:
        seed = args.seed + attempted
        attempted += 1
        runtime.policy = SO101PickPlaceExpert(env.kin, env)
        outcome = run_episode(runtime, seed, (recorder,))
        if args.sorting:
            recorder.set_task(f"put the {env.target_color} cube on the green pad")

        if outcome.success or args.keep_failures:
            path = recorder.end(outcome.success, seed)
            kept += 1
            print(
                f"[{kept}/{args.episodes}] seed={seed} "
                f"success={outcome.success} -> {path.name}"
            )
        else:
            recorder.discard()
            print(f"[--] seed={seed} failed, discarded")

        if attempted > args.episodes * 8:
            print("Giving up: the expert is failing too often. Check the task config.")
            break

    recorder.close()
    session.close()
    print(
        f"\nwrote {kept} episodes ({attempted} attempts) -> {args.dataset / 'meta.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
