"""Shared argparse flags reused across scripts/*.py.

Every script still owns its own defaults, help text, and choices where they
differ; these helpers only remove the repeated `add_argument` boilerplate
for flags that appear in more than one script.
"""

from __future__ import annotations

import argparse
from pathlib import Path


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


def _video_name(text: str) -> str:
    if not text or any(char in text for char in "/\\"):
        raise argparse.ArgumentTypeError(
            f"--video-name must be a plain file name prefix, got {text!r}"
        )
    return text


def add_video_name(parser: argparse.ArgumentParser) -> None:
    """`--video-name`: replace the automatic `<sim>_<robot>_<policy>` prefix."""
    parser.add_argument(
        "--video-name",
        type=_video_name,
        help="video file name prefix, replacing the automatic "
        "<simulator>_<robot>_<policy>; `_seed<seed>` and, on a repeat, `_02`, "
        "`_03`, ... are still added",
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
