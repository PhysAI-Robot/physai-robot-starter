"""Episode recorder in a LeRobot-shaped layout.

Writes one .npz per episode plus a dataset-level meta.json. Keys match the
LeRobot v2 convention so `scripts/to_lerobot.py` (or your own converter) is a
rename, not a reshape:

    observation.images.front  (T, H, W, 3) uint8
    observation.images.wrist  (T, H, W, 3) uint8
    observation.state         (T, 6)  float32  arm joints + gripper, radians
    action                    (T, 6)  float32  absolute joint targets, radians
    reward, done, phase       (T,)

Absolute joint targets — not deltas — because that is what the real SO-101
follower takes, so a policy trained here transfers to hardware without an
action-space change.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..contracts import Action, Observation


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
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.task = task
        self.fps = fps
        self.store_images = store_images
        self.episodes: list[dict] = []
        self._buf: EpisodeBuffer | None = None

    # ------------------------------------------------------------------
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
        """Append one control tick.

        `observation` is the state the policy saw; `action` is what it emitted
        in response — the standard (s, a) alignment for behaviour cloning.
        `gripper_joint` is the commanded gripper in *radians*; pass it so the
        recorded action is homogeneous with `observation.state`.
        """
        if self._buf is None:
            raise RuntimeError("call start_episode() first")

        if self.store_images:
            for name, frame in observation.images.items():
                self._buf.images.setdefault(name, []).append(frame.data.copy())

        self._buf.state.append(observation.joint_state.position.astype(np.float32))
        grip = gripper_joint if gripper_joint is not None else action.gripper.clipped()
        self._buf.action.append(
            np.concatenate([action.joint_position, [grip]]).astype(np.float32)
        )
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

        meta = {
            "index": idx,
            "file": path.name,
            "length": len(self._buf),
            "success": bool(success),
            "task": self.task,
            "fps": self.fps,
            **(extra or {}),
        }
        self.episodes.append(meta)
        self._buf = None
        return path

    def write_meta(self) -> Path:
        n_ok = sum(e["success"] for e in self.episodes)
        meta = {
            "codebase_version": "physai-0.0.1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "robot_type": "so101",
            "task": self.task,
            "fps": self.fps,
            "num_episodes": len(self.episodes),
            "num_successful_episodes": n_ok,
            "total_frames": sum(e["length"] for e in self.episodes),
            "features": {
                "observation.state": {"dtype": "float32", "shape": [6],
                                      "names": ["shoulder_pan", "shoulder_lift",
                                                "elbow_flex", "wrist_flex",
                                                "wrist_roll", "gripper"]},
                "action": {"dtype": "float32", "shape": [6],
                           "names": ["shoulder_pan", "shoulder_lift", "elbow_flex",
                                     "wrist_flex", "wrist_roll", "gripper"]},
            },
            "episodes": self.episodes,
        }
        path = self.root / "meta.json"
        path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return path


def load_episode(path: str | Path) -> dict:
    with np.load(Path(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
