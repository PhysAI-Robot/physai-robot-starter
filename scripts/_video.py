"""Episode video helpers shared by `run_sim.py` and `eval_policy.py`."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from physai.runtime import EpisodeObserver
from physai.sim.workspace import VIDEO_CAMERA

VIDEO_MODES = ("all", "failures", "successes")


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
    unsuccessful ones with `failures`, only successful ones with `successes`,
    none when recording is off."""
    if mode is None:
        return False
    return mode == "all" or (mode == "successes") == success


def default_video_name(
    simulator: str, robot: str, policy: str, renderer: str | None = None
) -> str:
    engines = simulator if renderer in (None, simulator) else f"{simulator}-{renderer}"
    return f"{engines}_{robot}_{policy}"


def next_video_stem(
    directory: Path,
    name: str,
    seed: int,
    suffixes: tuple[str, ...] = (".mp4", ".gif"),
    outcome: str = "",
) -> Path:
    """`<dir>/<name>_seed<seed>` for the first video of a seed, then `..._02`,
    `..._03` and so on, so a repeat run never overwrites an earlier video of
    the same seed. `name` is `default_video_name(...)` unless overridden.
    `outcome` (`_o` for a success, `_x` for a failure) ends the stem, and each
    outcome counts its own repeats."""
    stem = directory / f"{name}_seed{seed:04d}{outcome}"
    ext = "|".join(re.escape(suffix) for suffix in suffixes)
    repeat = re.compile(rf"{re.escape(stem.name)}_(\d{{2,}})(?:{ext})$")
    runs = [
        int(match.group(1))
        for path in directory.glob(f"{stem.name}_*")
        if (match := repeat.fullmatch(path.name))
    ]
    if any(stem.with_suffix(suffix).exists() for suffix in suffixes):
        runs.append(1)
    if not runs:
        return stem
    return directory / f"{stem.name}_{max(runs) + 1:02d}"


class VideoObserver(EpisodeObserver):
    """Collects one camera's frames over an episode, for `write_video`.

    The default is the scene's diagonal `video` view when the robot can draw it
    (`render_camera` and a scene with a `video_view`); it is rendered here on
    request, not read from the observation. Any observation camera can be
    named instead.
    """

    def __init__(self, runtime, camera: str | None = None) -> None:
        robot = runtime.robot
        has_view = getattr(getattr(robot, "cfg", None), "scene", None) is not None
        has_view = has_view and hasattr(robot.cfg.scene, "video_view")
        has_view = has_view and hasattr(robot, "render_camera")
        self._render = robot.render_camera if has_view else None
        self.camera = camera or (
            VIDEO_CAMERA if has_view else next(iter(runtime.robot_spec.camera_frames))
        )
        if self.camera == VIDEO_CAMERA and not has_view:
            raise ValueError("this robot has no `video` view; name a camera instead")
        self.frames: list[np.ndarray] = []

    def _grab(self, observation) -> None:
        if self.camera == VIDEO_CAMERA:
            self.frames.append(np.asarray(self._render(VIDEO_CAMERA)))
            return
        frame = observation.images.get(self.camera)
        if frame is not None:
            self.frames.append(np.asarray(frame.data))

    def on_reset(self, observation) -> None:
        self.frames.clear()
        self._grab(observation)

    def after_step(self, observation, action, next_observation, reward, done, info):
        self._grab(next_observation)
