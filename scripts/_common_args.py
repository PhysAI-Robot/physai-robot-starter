"""Shared argparse flags reused across scripts/*.py.

Every script still owns its own defaults, help text, and choices where they
differ; these helpers only remove the repeated `add_argument` boilerplate
for flags that appear in more than one script.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any

from _cli import add_robot  # noqa: F401  (stdlib-only, re-exported for the scripts)
from _video import VIDEO_MODES

# The SO-101 sessions every run script defaults to (and `--sorting` selects).
DEFAULT_MANIFEST = Path("configs/manifests/so101_single_cube_fixed_place.yaml")
SORTING_MANIFEST = Path("configs/manifests/so101_sorting.yaml")


def add_seed(
    parser: argparse.ArgumentParser, *, default: int | None = 0, help: str | None = None
) -> None:
    parser.add_argument(
        "--seed",
        type=int,
        default=default,
        help=help or "seed of the first episode; episode N uses seed + N",
    )


def add_episodes(
    parser: argparse.ArgumentParser, *, default: int, help: str | None = None
) -> None:
    parser.add_argument(
        "--episodes",
        type=int,
        default=default,
        help=help or "number of episodes to run",
    )


def add_max_steps(
    parser: argparse.ArgumentParser,
    *,
    default: int | None = None,
    help: str | None = None,
) -> None:
    parser.add_argument(
        "--max-steps",
        type=int,
        default=default,
        help=help or "episode length in control steps (default: the manifest's)",
    )


def add_checkpoint(parser: argparse.ArgumentParser, *, help: str | None = None) -> None:
    parser.add_argument(
        "--checkpoint", type=Path, help=help or "policy checkpoint directory"
    )


def parse_policy_arg(text: str) -> tuple[str, Any]:
    """Split `KEY=VALUE`, reading VALUE as a Python literal (a bare word stays a string)."""
    key, separator, raw = text.partition("=")
    if not separator or not key:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {text!r}")
    try:
        return key, ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return key, raw


def add_policy_args(parser: argparse.ArgumentParser) -> None:
    """`--checkpoint` and `--policy-arg`: the run-time inputs of a policy."""
    add_checkpoint(
        parser, help="policy checkpoint directory, for policies that load one"
    )
    parser.add_argument(
        "--policy-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        type=parse_policy_arg,
        help="override a policy constructor option, e.g. "
        "--policy-arg final_camera=front "
        "--policy-arg grasp_offset_xy='(0.0, 0.0)'; repeatable",
    )


def policy_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """The constructor arguments `--policy-arg` and `--checkpoint` give the policy."""
    kwargs = dict(args.policy_arg)
    if args.checkpoint is not None:
        kwargs["checkpoint"] = args.checkpoint
    return kwargs


def add_list_policies(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--list-policies",
        action="store_true",
        help="print the policies available for the manifest's robot (after its "
        "`plugins:` load) and exit",
    )


def print_policies(manifest_path: Path, *, exclude: tuple[str, ...] = ()) -> int:
    """List what `--policy` accepts for the robot in `manifest_path`, with its traits."""
    from physai.config import load_manifest
    from physai.policy import available_policies, policy_traits

    manifest = load_manifest(manifest_path)
    robot = manifest.robots[0].robot
    plugins = ", ".join(manifest.plugins) or "none"
    print(f"policies for robot {robot} (manifest: {manifest_path}, plugins: {plugins})")
    for name in available_policies(robot):
        if name in exclude:
            continue
        traits = policy_traits(name, robot)
        notes = f"simulators: {', '.join(traits.simulators)}"
        if traits.needs_images:
            notes += "; needs images"
        print(f"  {name:<16} {notes}")
    return 0


def add_out(
    parser: argparse.ArgumentParser, *, default: Path, help: str, metavar: str = "FILE"
) -> None:
    """`--out` as a single output file (scripts that write many files use
    `--out DIR`; see the CLI conventions in docs/ARCHITECTURE.md)."""
    parser.add_argument("--out", type=Path, default=default, metavar=metavar, help=help)


def add_policy(
    parser: argparse.ArgumentParser,
    *,
    default: str | None = None,
    choices=None,
    help: str | None = None,
) -> None:
    parser.add_argument(
        "--policy", default=default, choices=choices, help=help or "policy to run"
    )


def add_camera_resolution(parser: argparse.ArgumentParser) -> None:
    """`--camera-res`: one of the supported resolutions (default 320x240)."""
    from physai.contracts import CAMERA_RESOLUTIONS, DEFAULT_CAMERA_RESOLUTION

    parser.add_argument(
        "--camera-res",
        dest="camera_resolution",
        choices=CAMERA_RESOLUTIONS,
        default=None,
        help=f"camera render resolution (default: {DEFAULT_CAMERA_RESOLUTION}, "
        "or the manifest's simulation.camera_resolution)",
    )


def add_simulator(parser: argparse.ArgumentParser, *, help: str | None = None) -> None:
    """`--sim`: overrides the manifest's `simulator` field (default: mujoco)."""
    parser.add_argument(
        "--sim",
        dest="simulator",
        choices=("mujoco", "isaac"),
        default=None,
        help=help or "override the manifest's simulator engine",
    )


def _run_name(text: str) -> str:
    if not text or any(char in text for char in "/\\"):
        raise argparse.ArgumentTypeError(
            f"--name must be a plain file name prefix, got {text!r}"
        )
    return text


def add_run_outputs(parser: argparse.ArgumentParser) -> None:
    """What a run saves: a video, a data recording, and where and under what name."""
    parser.add_argument(
        "--video",
        nargs="?",
        const="all",
        choices=VIDEO_MODES,
        help="write a video per episode (`--video` or `--video all`), or only "
        "for the episodes that fail (`--video failures`)",
    )
    parser.add_argument(
        "--camera",
        help="camera that --video records (default: the robot's first camera)",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="record every input (cameras, joints, detections, grip force, ...) "
        "per episode as <out>/recordings/<name>_seed<seed>.npz (+ .json)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs"),
        help="directory the videos (videos/) and recordings (recordings/) go in",
    )
    parser.add_argument(
        "--name",
        type=_run_name,
        help="file name prefix for videos and recordings, replacing the automatic "
        "<simulator>_<robot>_<policy>; `_seed<seed>` and, on a repeat, `_02`, "
        "`_03`, ... are still added",
    )
