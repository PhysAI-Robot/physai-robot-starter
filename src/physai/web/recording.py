"""Browser-driven episode recording on top of `physai.data.EpisodeRecorder`.

`SessionRecorder` owns no episode format: it feeds the same recorder,
`.npz` layout and `meta.json` that `scripts/collect_demos.py` uses, plus the
optional `observation.environment_state` key (full simulator qpos) that lets
the viewer restore a recorded frame exactly. See docs/adr/0009.

The physics thread calls `record_tick()` while a browser-facing thread calls
`start()` / `stop()`. `_lock` guards the state `record_tick()` touches;
`_io_lock` serializes `start()` / `stop()` so a slow file write in `stop()`
never blocks the physics thread.
"""

from __future__ import annotations

import dataclasses
import json
import threading
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import Action, Observation
from ..data import EpisodeRecorder, load_episode

_STATE_KEY = "observation.environment_state"


class SessionRecorder:
    def __init__(
        self,
        root: str | Path,
        *,
        robot: Any,
        fps: float,
        environment_state_dim: int | None,
    ) -> None:
        spec = robot.robot_spec
        self._camera_names = tuple(spec.camera_frames)
        self._gripper_to_joint = getattr(robot, "gripper_to_joint", None)
        self._recorder = EpisodeRecorder(
            root,
            task="teleoperation",
            fps=fps,
            store_images=bool(self._camera_names),
            robot_spec=spec,
            training_contract=getattr(robot, "training_contract", None),
            simulator_config={"control_hz": fps},
            environment_state_dim=environment_state_dim,
        )
        self._resume_existing(environment_state_dim)
        self._lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._active = False
        self._frames = 0
        self._discarded = 0
        self._error: str | None = None
        self._waiting_for: list[str] = []

    def _resume_existing(self, environment_state_dim: int | None) -> None:
        """Continue an existing dataset instead of overwriting episode_00000."""
        meta_path = self._recorder.root / "meta.json"
        if not meta_path.exists():
            return
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("robot_type") != self._recorder.robot_type:
            raise ValueError(
                f"{meta_path} was recorded for robot {meta.get('robot_type')!r}, "
                f"not {self._recorder.robot_type!r}; use a fresh --record-dir"
            )
        existing_state = meta.get("features", {}).get(_STATE_KEY)
        expected_shape = (
            None if environment_state_dim is None else [environment_state_dim]
        )
        if (existing_state or {}).get("shape") != expected_shape:
            raise ValueError(
                f"{meta_path} has a different {_STATE_KEY} layout; "
                "use a fresh --record-dir"
            )
        self._recorder.episodes = list(meta["episodes"])

    @property
    def active(self) -> bool:
        return self._active

    def episodes(self) -> list[dict[str, Any]]:
        """Saved episodes; `playable` is true when they carry the qpos to restore."""
        playable = self._recorder.environment_state_dim is not None
        with self._lock:
            return [
                {**episode, "playable": playable} for episode in self._recorder.episodes
            ]

    def load_states(self, file: str) -> tuple[np.ndarray, dict[str, Any]]:
        """The `(T, nq)` recorded simulator states of one saved episode.

        Only episodes this dataset lists are loadable, so a client can never
        make the host open an arbitrary path.
        """
        entry = next((e for e in self.episodes() if e["file"] == file), None)
        if entry is None:
            raise ValueError(f"unknown episode {file!r}")
        if not entry["playable"]:
            raise ValueError(f"episode {file!r} has no {_STATE_KEY} to play back")
        try:
            states = load_episode(self._recorder.root / file, keys=(_STATE_KEY,))[
                _STATE_KEY
            ]
        except (OSError, KeyError, zipfile.BadZipFile) as exc:
            raise ValueError(f"cannot read episode {file!r}: {exc}") from exc
        return states, entry

    def status(self) -> dict[str, Any]:
        with self._lock:
            episodes = list(self._recorder.episodes)
            return {
                "enabled": True,
                "active": self._active,
                "frames": self._frames,
                "episodes_saved": len(episodes),
                "successes": sum(bool(e["success"]) for e in episodes),
                "discarded": self._discarded,
                "dir": str(self._recorder.root),
                "waiting_for": list(self._waiting_for),
                "error": self._error,
            }

    def start(self) -> None:
        with self._io_lock, self._lock:
            if self._active:
                raise RuntimeError("already recording")
            self._recorder.start_episode()
            self._active = True
            self._frames = 0
            self._error = None
            self._waiting_for = []

    def record_tick(
        self,
        observation: Observation,
        action: Action,
        environment_state: np.ndarray | None,
    ) -> None:
        """Record one step; a bad frame aborts the take instead of raising.

        This runs on the physics thread, which must survive a recording
        problem: the take is dropped and the reason is surfaced in `status()`.
        Steps are skipped (not recorded) until every camera has produced a
        frame, so a take never holds steps with missing images.
        """
        with self._lock:
            if not self._active:
                return
            self._waiting_for = [
                name for name in self._camera_names if name not in observation.images
            ]
            if self._waiting_for:
                return
            try:
                images = {name: observation.images[name] for name in self._camera_names}
                gripper_joint = None
                if self._gripper_to_joint is not None and action.gripper is not None:
                    gripper_joint = float(
                        self._gripper_to_joint(action.gripper.clipped())
                    )
                self._recorder.record(
                    dataclasses.replace(observation, images=images),
                    action,
                    gripper_joint=gripper_joint,
                    environment_state=environment_state,
                )
                self._frames += 1
            except ValueError as exc:
                self._drop_take_locked(f"recording aborted: {exc}")

    def stop(self, success: bool | None) -> Path | None:
        """End the take: tag it success/fail, or discard it with `None`."""
        with self._io_lock:
            with self._lock:
                if not self._active:
                    raise RuntimeError("not recording")
                self._active = False
                if success is None:
                    self._drop_take_locked(None)
                    return None
            # record_tick() ignores frames while inactive, so the buffer is
            # ours alone and the slow compressed write stays off `_lock`.
            path = self._recorder.end_episode(success, extra={"source": "web"})
            if path is None:
                raise RuntimeError("no frames were recorded; nothing was saved")
            self._recorder.write_meta()
            return path

    def abort(self, reason: str | None = None) -> None:
        """Drop an in-progress take (e.g. the world was reset under it)."""
        with self._lock:
            if self._active:
                self._drop_take_locked(reason)

    def _drop_take_locked(self, reason: str | None) -> None:
        self._active = False
        self._recorder.discard_episode()
        self._discarded += 1
        self._error = reason
