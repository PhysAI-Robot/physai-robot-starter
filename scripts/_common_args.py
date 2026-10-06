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

from _video import VIDEO_MODES

# The SO-101 sessions every run script defaults to (and `--sorting` selects).
DEFAULT_MANIFEST = Path("configs/manifests/so101_single_cube_fixed_place.yaml")
SORTING_MANIFEST = Path("configs/manifests/so101_sorting.yaml")


def add_seed(
    parser: argparse.ArgumentParser, *, default: int | None = 0, help: str | None = None
) -> None:
    parser.add_argument("--seed", type=int, default=default, help=help)


def add_episodes(
    parser: argparse.ArgumentParser, *, default: int, help: str | None = None
) -> None:
    parser.add_argument("--episodes", type=int, default=default, help=help)


def add_max_steps(
    parser: argparse.ArgumentParser,
    *,
    default: int | None = None,
    help: str | None = None,
) -> None:
    parser.add_argument("--max-steps", type=int, default=default, help=help)


def add_checkpoint(parser: argparse.ArgumentParser, *, help: str | None = None) -> None:
    parser.add_argument("--checkpoint", type=Path, help=help)


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
    add_checkpoint(parser, help="required for --policy lerobot")
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


def policy_kwargs(args: argparse.Namespace, policy: str) -> dict[str, Any]:
    """The constructor arguments `--policy-arg` and `--checkpoint` give `policy`."""
    kwargs = dict(args.policy_arg)
    if policy == "lerobot":
        if args.checkpoint is None:
            raise ValueError("--policy lerobot needs --checkpoint")
        kwargs["checkpoint"] = args.checkpoint
    return kwargs


def add_out(
    parser: argparse.ArgumentParser, *, default: Path, help: str | None = None
) -> None:
    parser.add_argument("--out", type=Path, default=default, help=help)


def add_robot(
    parser: argparse.ArgumentParser,
    *,
    default: str | None = None,
    choices=None,
    help: str | None = None,
) -> None:
    parser.add_argument("--robot", default=default, choices=choices, help=help)


def add_policy(
    parser: argparse.ArgumentParser,
    *,
    default: str | None = None,
    choices=None,
    help: str | None = None,
) -> None:
    parser.add_argument("--policy", default=default, choices=choices, help=help)


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
        "per episode as <out-dir>/recordings/<name>_seed<seed>.npz (+ .json)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="record into this dataset directory instead (episode_NNNNN.npz + "
        "meta.json, the layout training and --policy replay read; an existing "
        "dataset there is continued); implies --record",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs"),
        help="where videos (videos/) and recordings (recordings/) are written "
        "(default: outputs)",
    )
    parser.add_argument(
        "--name",
        type=_run_name,
        help="file name prefix for videos and recordings, replacing the automatic "
        "<simulator>_<robot>_<policy>; `_seed<seed>` and, on a repeat, `_02`, "
        "`_03`, ... are still added",
    )
