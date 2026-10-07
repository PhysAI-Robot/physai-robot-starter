import json

import numpy as np
import pytest

pytest.importorskip("torch")


def _write_dataset(root, episodes=3, steps=5, size=8):
    entries = []
    for index in range(episodes):
        name = f"episode_{index:05d}.npz"
        np.savez(
            root / name,
            **{
                "observation.state": np.zeros((steps, 6), dtype=np.float32),
                "action": np.zeros((steps, 6), dtype=np.float32),
                "observation.images.front": np.zeros(
                    (steps, size, size, 3), dtype=np.uint8
                ),
            },
        )
        entries.append({"file": name, "seed": index})
    (root / "meta.json").write_text(
        json.dumps(
            {
                "episodes": entries,
                "observation_schema": {"observation.images.front": {}},
            }
        ),
        encoding="utf-8",
    )


def test_max_episodes_keeps_the_first_n_episodes(tmp_path):
    from research.imitation_learning.act_dataset import ACTEpisodeDataset

    _write_dataset(tmp_path)
    everything = ACTEpisodeDataset(tmp_path, image_size=8)
    first_two = ACTEpisodeDataset(tmp_path, image_size=8, max_episodes=2)
    assert len(everything.episodes) == 3 and len(everything) == 15
    assert len(first_two.episodes) == 2 and len(first_two) == 10
