"""State of one recorded episode being replayed through the paused host.

`Playback` is plain state; `Host` owns the physics lock and applies it. See
docs/adr/0009: playback restores recorded simulator state on the paused world
rather than re-simulating, and exiting restores the world it entered from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..contracts import Action

MAX_SPEED = 8.0


@dataclass
class Playback:
    file: str
    states: np.ndarray  # (T, nq) recorded simulator qpos
    fps: float
    success: bool | None
    # The live world this playback interrupted, restored by exit_playback().
    live_data: Any
    live_observation: Any
    live_hold_action: Action | None
    position: float = 0.0  # fractional frame index, so 0.5x still advances
    playing: bool = False
    speed: float = 1.0
    shown: int = -1  # last frame written to the simulator

    @property
    def length(self) -> int:
        return len(self.states)

    @property
    def frame(self) -> int:
        # The epsilon keeps 1.0-per-tick accumulation from landing on 0.999...
        return min(int(self.position + 1e-6), self.length - 1)

    def seek(self, frame: int) -> None:
        self.position = float(min(max(int(frame), 0), self.length - 1))

    def advance(self, frames: float) -> None:
        """Move forward; stops at the last frame rather than looping."""
        self.position += frames
        if self.position >= self.length - 1:
            self.position = float(self.length - 1)
            self.playing = False

    def set_speed(self, speed: float) -> None:
        if not 0.0 < speed <= MAX_SPEED:
            raise ValueError(f"playback speed must be in (0, {MAX_SPEED:g}]")
        self.speed = float(speed)

    def status(self) -> dict[str, Any]:
        return {
            "active": True,
            "file": self.file,
            "frame": self.frame,
            "length": self.length,
            "playing": self.playing,
            "speed": self.speed,
            "success": self.success,
        }
