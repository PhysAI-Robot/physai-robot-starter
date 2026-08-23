"""Episode recorder in a LeRobot-shaped layout."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..contracts import Action, Observation
from ..robots.base import RobotSpec


@dataclass
class EpisodeBuffer:
    images: dict[str, list] = field(default_factory=dict)
    state: list = field(default_factory=list)
    action: list = field(default_factory=list)
    reward: list = field(default_factory=list)
    done: list = field(default_factory=list)
    phase: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.state)


class EpisodeRecorder:
    def __init__(
        self,
        root: str | Path,
        task: str = "put the red cube on the green pad",
        fps: float = 25.0,
        store_images: bool = True,
        robot_type: str = "so101",
        robot_spec: RobotSpec | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.task = task
        self.fps = fps
        self.store_images = store_images
        self.robot_spec = robot_spec
        self.robot_type = robot_spec.name if robot_spec else robot_type
        self._state_names: tuple[str, ...] | None = None
        self._action_names: tuple[str, ...] | None = None
        self.episodes: list[dict] = []
        self._buf: EpisodeBuffer | None = None

    def start_episode(self) -> None:
        self._buf = EpisodeBuffer()

    def record(
        self,
        observation: Observation,
        action: Action,
        reward: float = 0.0,
        done: bool = False,
        phase: str = "",
        gripper_joint: float | None = None,
    ) -> None:
        if self._buf is None:
            raise RuntimeError("call start_episode() first")
        if action.joint_position is not None:
            action_values = action.joint_position
            action_names = tuple(
                f"joint_{index}" for index in range(action_values.size)
            ) + ("gripper",)
        elif action.ee_twist is not None:
            action_values = action.ee_twist.as_array()
            action_names = (
                "linear_x", "linear_y", "linear_z",
                "angular_x", "angular_y", "angular_z",
            )
        else:
            raise ValueError("EpisodeRecorder requires joint-position or twist actions")

        self._state_names = self._state_names or observation.joint_state.name
        self._action_names = self._action_names or action_names

        if self.store_images:
            for name, frame in observation.images.items():
                self._buf.images.setdefault(name, []).append(frame.data.copy())

        self._buf.state.append(observation.joint_state.position.astype(np.float32))
        if action.joint_position is not None:
            grip = gripper_joint if gripper_joint is not None else action.gripper.clipped()
            action_values = np.concatenate([action_values, [grip]])
        self._buf.action.append(action_values.astype(np.float32))
        self._buf.reward.append(float(reward))
        self._buf.done.append(bool(done))
        self._buf.phase.append(phase)

    def end_episode(self, success: bool, extra: dict | None = None) -> Path | None:
        if self._buf is None or len(self._buf) == 0:
            self._buf = None
            return None

        idx = len(self.episodes)
        path = self.root / f"episode_{idx:05d}.npz"
        arrays = {
            "observation.state": np.stack(self._buf.state),
            "action": np.stack(self._buf.action),
            "reward": np.asarray(self._buf.reward, dtype=np.float32),
            "done": np.asarray(self._buf.done, dtype=bool),
            "phase": np.asarray(self._buf.phase),
        }
        for name, frames in self._buf.images.items():
            arrays[f"observation.images.{name}"] = np.stack(frames)
        np.savez_compressed(path, **arrays)

        self.episodes.append({
            "index": idx,
            "file": path.name,
            "length": len(self._buf),
            "success": bool(success),
            "task": self.task,
            "fps": self.fps,
            **(extra or {}),
        })
        self._buf = None
        return path

    def write_meta(self) -> Path:
        n_ok = sum(e["success"] for e in self.episodes)
        state_names = list(self._state_names or ())
        action_names = list(self._action_names or ())
        if self.robot_spec is not None:
            state_names = list(self.robot_spec.joint_names)
        if self.robot_type == "so101" and self.robot_spec is None:
            state_names = [
                "shoulder_pan", "shoulder_lift", "elbow_flex",
                "wrist_flex", "wrist_roll", "gripper",
            ]
            action_names = state_names.copy()
        meta = {
            "codebase_version": "physai-0.0.1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "robot_type": self.robot_type,
            "task": self.task,
            "fps": self.fps,
            "num_episodes": len(self.episodes),
            "num_successful_episodes": n_ok,
            "total_frames": sum(e["length"] for e in self.episodes),
            "features": {
                "observation.state": {"dtype": "float32", "shape": [len(state_names)], "names": state_names},
                "action": {"dtype": "float32", "shape": [len(action_names)], "names": action_names},
            },
            "episodes": self.episodes,
        }
        path = self.root / "meta.json"
        path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return path


def load_episode(path: str | Path) -> dict:
    with np.load(Path(path), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}
