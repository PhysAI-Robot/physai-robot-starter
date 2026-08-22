"""Turns recorded .npz episodes into ACT training batches.

Deliberately bypasses LeRobot's on-disk `LeRobotDataset` format (which encodes
episodes as video via a system `ffmpeg` binary — fragile on a bare Windows
install). Episodes already sit in memory as numpy arrays in the exact key
layout ACT expects (`data/recorder.py` was written to match), so this reads
them directly into a `torch.utils.data.Dataset`. The actual model
(`ACTPolicy`) and its pre/post-processing pipeline are the real LeRobot
library code — only the on-disk packaging is swapped out.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.recorder import load_episode


@dataclass
class DatasetStats:
    """mean/std per feature key, in the shape `make_act_pre_post_processors` expects."""

    per_key: dict[str, dict[str, list[float]]]

    def to_json(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.per_key, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "DatasetStats":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


class ACTEpisodeDataset(Dataset):
    """One sample = one timestep, paired with the next `chunk_size` actions.

    Images are resized to `image_size` and returned as CHW float32 in [0, 1].
    Padded chunk positions (past the end of an episode) are marked in
    `action_is_pad` and filled with the episode's final action, matching what
    `ACTPolicy.forward` expects (it masks padded positions out of the loss).
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        camera_keys: tuple[str, ...] = ("front", "wrist"),
        chunk_size: int = 30,
        image_size: int = 128,
        task: str = "put the red cube on the green pad",
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.camera_keys = camera_keys
        self.chunk_size = chunk_size
        self.image_size = image_size
        self.task = task

        meta = json.loads((self.dataset_dir / "meta.json").read_text(encoding="utf-8"))
        self.episodes: list[dict[str, np.ndarray]] = []
        self.index: list[tuple[int, int]] = []  # (episode_idx, timestep)
        for e in meta["episodes"]:
            data = load_episode(self.dataset_dir / e["file"])
            data = {k: np.asarray(v) for k, v in data.items()}
            self.episodes.append(data)
            ep_idx = len(self.episodes) - 1
            T = data["observation.state"].shape[0]
            self.index.extend((ep_idx, t) for t in range(T))

        if not self.episodes:
            raise ValueError(f"no episodes found under {self.dataset_dir}")

    def __len__(self) -> int:
        return len(self.index)

    def _image(self, arr: np.ndarray) -> torch.Tensor:
        """(H, W, 3) uint8 -> (3, image_size, image_size) float32 in [0, 1].

        Centre-crop before resize so a non-square source (recorded with
        different --width/--height) degrades gracefully instead of being
        stretched — see the matching note in vla_adapter.LeRobotPolicy._resize,
        which a mismatch here would silently be inconsistent with at eval time.
        """
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        if t.shape[-2] != t.shape[-1]:
            h, w = t.shape[-2], t.shape[-1]
            side = min(h, w)
            top, left = (h - side) // 2, (w - side) // 2
            t = t[:, top:top + side, left:left + side]
        if t.shape[-1] != self.image_size:
            t = torch.nn.functional.interpolate(
                t.unsqueeze(0), size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
        return t

    def __getitem__(self, i: int) -> dict:
        ep_idx, t = self.index[i]
        ep = self.episodes[ep_idx]
        T = ep["observation.state"].shape[0]

        actions = ep["action"]
        end = min(t + self.chunk_size, T)
        chunk = actions[t:end]
        n_pad = self.chunk_size - chunk.shape[0]
        is_pad = np.zeros(self.chunk_size, dtype=bool)
        if n_pad > 0:
            pad = np.repeat(actions[T - 1:T], n_pad, axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
            is_pad[-n_pad:] = True

        sample = {
            "observation.state": torch.from_numpy(ep["observation.state"][t]).float(),
            "action": torch.from_numpy(chunk).float(),
            "action_is_pad": torch.from_numpy(is_pad),
            "task": self.task,
        }
        for cam in self.camera_keys:
            key = f"observation.images.{cam}"
            sample[key] = self._image(ep[key][t])
        return sample

    def compute_stats(self) -> DatasetStats:
        """Mean/std over every frame in the dataset (not just this dataloader's batches)."""
        state = np.concatenate([e["observation.state"] for e in self.episodes], axis=0)
        action = np.concatenate([e["action"] for e in self.episodes], axis=0)
        per_key = {
            "observation.state": {
                "mean": state.mean(0).tolist(), "std": (state.std(0) + 1e-6).tolist(),
            },
            "action": {
                "mean": action.mean(0).tolist(), "std": (action.std(0) + 1e-6).tolist(),
            },
        }
        for cam in self.camera_keys:
            key = f"observation.images.{cam}"
            # Sample frames rather than decoding every one at full res — image
            # normalization only needs a stable per-channel estimate.
            n_ep = len(self.episodes)
            sample_frames = []
            for e in self.episodes:
                frames = e[key]
                idx = np.linspace(0, frames.shape[0] - 1, num=min(8, frames.shape[0])).astype(int)
                sample_frames.append(frames[idx].astype(np.float32) / 255.0)
            stacked = np.concatenate(sample_frames, axis=0)  # (N, H, W, 3)
            mean = stacked.mean(axis=(0, 1, 2))
            std = stacked.std(axis=(0, 1, 2)) + 1e-6
            per_key[key] = {"mean": mean.tolist(), "std": std.tolist()}
        return DatasetStats(per_key)
