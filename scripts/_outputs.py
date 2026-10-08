"""What `run_sim.py` and `eval_policy.py` save per episode, from one set of flags.

`--video` (a camera's frames) and `--record` (every input as
data) both watch the episode through `EpisodeObserver`s; this owns creating
them, the file names, and writing them once the episode has run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from _recording import RunRecorder
from _video import (
    VideoObserver,
    default_video_name,
    keep_video,
    next_video_stem,
    write_video,
)


def wants_cameras(args: argparse.Namespace) -> bool:
    """Whether the flags need camera frames rendered."""
    return args.video is not None or args.record


class RunOutputs:
    def __init__(
        self,
        runtime,
        args: argparse.Namespace,
        *,
        simulator: str,
        robot: str,
        policy: str,
        task: str,
        fps: float,
        dataset_dir: Path | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._runtime, self._args, self._fps = runtime, args, fps
        self._name = args.name or default_video_name(simulator, robot, policy)
        self._video: VideoObserver | None = None
        self.recorder = (
            RunRecorder(
                runtime,
                fps=fps,
                name=self._name,
                task=task,
                dataset_dir=dataset_dir,
                out_dir=args.out,
                metadata=metadata,
            )
            if args.record or dataset_dir is not None
            else None
        )

    def begin(self) -> tuple:
        """The observers for the next episode."""
        self._video = (
            VideoObserver(self._runtime, self._args.camera)
            if self._args.video is not None
            else None
        )
        return tuple(item for item in (self._video, self.recorder) if item is not None)

    def save(self, success: bool, seed: int) -> None:
        """Write what the episode just run produced, and say where."""
        if self.recorder:
            saved = self.recorder.end(success, seed)
            if saved:
                print(f"  record -> {saved}")
        video = self._video
        if video and video.frames and keep_video(self._args.video, success):
            path = write_video(
                np.stack(video.frames),
                next_video_stem(self._args.out / "videos", self._name, seed),
                fps=int(self._fps),
            )
            print(f"  video -> {path}")

    def close(self) -> None:
        if self.recorder:
            self.recorder.close()
