"""Turns recorded .npz episodes into ACT training batches.

Deliberately bypasses LeRobot's on-disk `LeRobotDataset` format (which encodes
episodes as video via a system `ffmpeg` binary — fragile on a bare Windows
install). Episodes already sit in memory as numpy arrays in the exact key
layout ACT expects (`data/recorder.py` was written to match), so this reads
them directly into a `torch.utils.data.Dataset`. The actual model
(`ACTPolicy`) and its pre/post-processing pipeline are the real LeRobot
library code — only the on-disk packaging is swapped out.

Research module (imitation learning): imports `torch` at module scope, so
only `research/imitation_learning/train_act.py` imports this directly. Core
never imports this module (see research/README.md).
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from physai.data.recorder import load_episode


@dataclass
class DatasetStats:
    """mean/std per feature key, in the shape `make_act_pre_post_processors` expects."""

    per_key: dict[str, dict[str, list[float]]]

    def to_json(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.per_key, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> DatasetStats:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


class ACTEpisodeDataset(Dataset):
    """One sample = one timestep, paired with the next `chunk_size` actions.

    Images are resized to `image_size` and returned as CHW float32 in [0, 1].
    Padded chunk positions (past the end of an episode) are marked in
    `action_is_pad` and filled with the episode's final action, matching what
    `ACTPolicy.forward` expects (it masks padded positions out of the loss).

    `dataset_dir` is one dataset folder or several (their episodes are joined, e.g.
    the same demonstrations recorded in two simulators). The policy-sized frames are
    resized once, saved next to each dataset in `.act_cache_<image_size>/` and read
    back memory-mapped, so a dataset costs disk and page cache instead of RAM and
    pickles to DataLoader workers as a handful of paths.
    """

    def __init__(
        self,
        dataset_dir: str | Path | Sequence[str | Path],
        camera_keys: tuple[str, ...] | None = None,
        chunk_size: int = 30,
        image_size: int = 128,
        task: str | None = None,
        max_episodes: int | None = None,
        color_jitter: bool = False,
    ) -> None:
        single = isinstance(dataset_dir, (str, Path))
        dirs = [Path(d) for d in ([dataset_dir] if single else dataset_dir)]
        self.dataset_dirs = dirs
        self.dataset_dir = dirs[0]
        self.chunk_size = chunk_size
        self.image_size = image_size
        # Random brightness, contrast, saturation, a slight hue shift and a vertical
        # brightness ramp per camera and sample, training only. Isaac Sim renders
        # the front camera's background about 0.5x as bright as MuJoCo but its table
        # only about 0.85x (scripts/compare_cameras.py), which a global brightness
        # change cannot express; the ramp scales the top of the image apart from the
        # bottom.
        self._jitter = (
            v2.ColorJitter(
                brightness=(0.4, 1.4),
                contrast=(0.7, 1.3),
                saturation=(0.7, 1.3),
                hue=(-0.03, 0.03),
            )
            if color_jitter
            else None
        )

        metas = [
            json.loads((d / "meta.json").read_text(encoding="utf-8")) for d in dirs
        ]
        meta = metas[0]
        if camera_keys is None:
            camera_keys = tuple(
                key.removeprefix("observation.images.")
                for key in meta.get("observation_schema", {})
                if key.startswith("observation.images.")
            )
        self.camera_keys = camera_keys
        self.task = task if task is not None else meta.get("task", "")
        self.episodes: list[dict] = []
        self.index: list[tuple[int, int]] = []  # (episode_idx, timestep)
        self._frames: dict[tuple[int, str], np.ndarray] = {}
        # The first `max_episodes` recorded episodes of each folder, so one dataset
        # serves several demonstration counts.
        for folder, folder_meta in zip(dirs, metas):
            for e in folder_meta["episodes"][:max_episodes]:
                self.episodes.append(self._episode(folder, e["file"]))
                steps = self.episodes[-1]["observation.state"].shape[0]
                ep_idx = len(self.episodes) - 1
                self.index.extend((ep_idx, t) for t in range(steps))

        if not self.episodes:
            raise ValueError(f"no episodes found under {dirs}")

    def _episode(self, folder: Path, file: str) -> dict:
        """State, actions and the cached policy-sized frame files of one episode."""
        cache = folder / f".act_cache_{self.image_size}"
        stem = Path(file).stem
        paths = {cam: cache / f"{stem}_{cam}.npy" for cam in self.camera_keys}
        if not all(path.exists() for path in paths.values()):
            cache.mkdir(exist_ok=True)
            full = load_episode(folder / file)
            for cam, path in paths.items():
                if not path.exists():
                    shrunk = self._shrink(np.asarray(full[f"observation.images.{cam}"]))
                    tmp = path.with_name(path.name + ".tmp")
                    with tmp.open("wb") as handle:
                        np.save(handle, shrunk)
                    os.replace(tmp, path)
        data = load_episode(folder / file, keys=("observation.state", "action"))
        episode = {k: np.asarray(v) for k, v in data.items()}
        episode["frames"] = paths
        return episode

    def _image_at(self, ep_idx: int, cam: str, t: int) -> np.ndarray:
        """One policy-sized frame, from the memory-mapped cache (opened on first use)."""
        key = (ep_idx, cam)
        if key not in self._frames:
            path = self.episodes[ep_idx]["frames"][cam]
            self._frames[key] = np.load(path, mmap_mode="r")
        return self._frames[key][t]

    def __getstate__(self) -> dict:
        # memory maps are reopened in each DataLoader worker, not pickled
        state = self.__dict__.copy()
        state["_frames"] = {}
        return state

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
            t = t[:, top : top + side, left : left + side]
        if t.shape[-1] != self.image_size:
            t = torch.nn.functional.interpolate(
                t.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return t

    @staticmethod
    def _brightness_ramp(image: torch.Tensor) -> torch.Tensor:
        """Scale rows linearly from a random top gain to a random bottom gain."""
        top = float(torch.empty(1).uniform_(0.45, 1.1))
        bottom = float(torch.empty(1).uniform_(0.8, 1.15))
        gain = torch.linspace(top, bottom, image.shape[-2]).view(1, -1, 1)
        return image * gain

    def _shrink(self, frames: np.ndarray) -> np.ndarray:
        """(T, H, W, 3) uint8 -> (T, 3, image_size, image_size) uint8."""
        return np.stack(
            [(self._image(f) * 255.0).round().byte().numpy() for f in frames]
        )

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
            pad = np.repeat(actions[T - 1 : T], n_pad, axis=0)
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
            frame = np.array(self._image_at(ep_idx, cam, t))
            image = torch.from_numpy(frame).float() / 255.0
            if self._jitter is not None:
                image = self._brightness_ramp(self._jitter(image)).clamp(0.0, 1.0)
            sample[key] = image
        return sample

    def compute_stats(self) -> DatasetStats:
        """Mean/std over every frame in the dataset (not just this dataloader's batches)."""
        state = np.concatenate([e["observation.state"] for e in self.episodes], axis=0)
        action = np.concatenate([e["action"] for e in self.episodes], axis=0)
        per_key = {
            "observation.state": {
                "mean": state.mean(0).tolist(),
                "std": (state.std(0) + 1e-6).tolist(),
            },
            "action": {
                "mean": action.mean(0).tolist(),
                "std": (action.std(0) + 1e-6).tolist(),
            },
        }
        for cam in self.camera_keys:
            key = f"observation.images.{cam}"
            # Sample frames rather than decoding every one at full res — image
            # normalization only needs a stable per-channel estimate.
            sample_frames = []
            for ep_idx, e in enumerate(self.episodes):
                steps = e["observation.state"].shape[0]
                idx = np.linspace(0, steps - 1, num=min(8, steps)).astype(int)
                frames = np.stack([self._image_at(ep_idx, cam, t) for t in idx])
                sample_frames.append(frames.astype(np.float32) / 255.0)
            stacked = np.concatenate(sample_frames, axis=0)  # (N, 3, S, S)
            mean = stacked.mean(axis=(0, 2, 3))
            std = stacked.std(axis=(0, 2, 3)) + 1e-6
            per_key[key] = {"mean": mean.tolist(), "std": std.tolist()}
        return DatasetStats(per_key)
