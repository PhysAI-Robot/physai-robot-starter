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
