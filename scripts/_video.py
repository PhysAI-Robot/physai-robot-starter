"""Episode video helpers shared by `run_sim.py` and `eval_policy.py`."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

VIDEO_MODES = ("all", "failures")


def write_video(frames: np.ndarray, stem: Path, fps: int) -> Path:
    """Write mp4 if an H.264 encoder is available, otherwise fall back to GIF.

    imageio's default pyav path raises an unhelpful `expected bytes, NoneType`
    when no codec is registered, so the codec is named explicitly and the
    fallback is silent-but-reported rather than a stack trace.
    """
    import imageio.v3 as iio

    stem.parent.mkdir(parents=True, exist_ok=True)
    mp4 = stem.with_suffix(".mp4")
    for plugin, kwargs in (
        ("FFMPEG", {"codec": "libx264"}),
        ("pyav", {"codec": "libx264"}),
    ):
        try:
            iio.imwrite(mp4, frames, fps=fps, plugin=plugin, **kwargs)
            return mp4
        except (ImportError, OSError, RuntimeError, TypeError, ValueError):
            continue

    gif = stem.with_suffix(".gif")
    iio.imwrite(gif, frames[::2], duration=2000 / fps, loop=0)
    print("  (no H.264 encoder found — wrote a GIF; run `uv sync` for mp4 support)")
    return gif


def keep_video(mode: str | None, success: bool) -> bool:
    """Whether an episode's video is written: every episode with `all`, only
    unsuccessful ones with `failures`, none when recording is off."""
    if mode is None:
        return False
    return mode == "all" or not success


def default_video_name(simulator: str, robot: str, policy: str) -> str:
    return f"{simulator}_{robot}_{policy}"


def next_video_stem(directory: Path, name: str, seed: int) -> Path:
    """`<dir>/<name>_seed<seed>` for the first video of a seed, then `..._02`,
    `..._03` and so on, so a repeat run never overwrites an earlier video of
    the same seed. `name` is `default_video_name(...)` unless overridden."""
    stem = directory / f"{name}_seed{seed:04d}"
    repeat = re.compile(rf"{re.escape(stem.name)}_(\d{{2,}})\.(?:mp4|gif)$")
    runs = [
        int(match.group(1))
        for path in directory.glob(f"{stem.name}_*")
        if (match := repeat.fullmatch(path.name))
    ]
    if any(stem.with_suffix(suffix).exists() for suffix in (".mp4", ".gif")):
        runs.append(1)
    if not runs:
        return stem
    return directory / f"{stem.name}_{max(runs) + 1:02d}"
