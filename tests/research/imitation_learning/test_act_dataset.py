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


def test_color_jitter_changes_training_images_only_when_asked(tmp_path):
    import torch

    from research.imitation_learning.act_dataset import ACTEpisodeDataset

    _write_dataset(tmp_path)
    # a mid-grey frame, so brightness changes in either direction stay in range
    for index in range(3):
        path = tmp_path / f"episode_{index:05d}.npz"
        data = dict(np.load(path))
        data["observation.images.front"][:] = 128
        np.savez(path, **data)
    plain = ACTEpisodeDataset(tmp_path, image_size=8)
    jittered = ACTEpisodeDataset(tmp_path, image_size=8, color_jitter=True)
    key = "observation.images.front"
    torch.manual_seed(0)
    changed = jittered[0][key]
    assert not torch.allclose(changed, plain[0][key])
    assert changed.min() >= 0.0 and changed.max() <= 1.0
    assert torch.equal(plain[0][key], plain[1][key])


def test_max_episodes_keeps_the_first_n_episodes(tmp_path):
    from research.imitation_learning.act_dataset import ACTEpisodeDataset

    _write_dataset(tmp_path)
    everything = ACTEpisodeDataset(tmp_path, image_size=8)
    first_two = ACTEpisodeDataset(tmp_path, image_size=8, max_episodes=2)
    assert len(everything.episodes) == 3 and len(everything) == 15
    assert len(first_two.episodes) == 2 and len(first_two) == 10


def test_several_folders_are_joined_and_frames_are_cached_on_disk(tmp_path):
    import pickle

    import torch

    from research.imitation_learning.act_dataset import ACTEpisodeDataset

    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    _write_dataset(first, episodes=2)
    _write_dataset(second, episodes=3)
    joined = ACTEpisodeDataset([first, second], image_size=8)
    assert len(joined.episodes) == 5 and len(joined) == 25
    # `max_episodes` counts per folder
    assert (
        len(ACTEpisodeDataset([first, second], image_size=8, max_episodes=1).episodes)
        == 2
    )
    cache = first / ".act_cache_8"
    assert len(list(cache.glob("*.npy"))) == 2  # one front-camera file per episode
    # the cache is reused, and a pickled copy (a DataLoader worker) carries no frames
    again = ACTEpisodeDataset(first, image_size=8)
    assert torch.equal(
        again[0]["observation.images.front"], joined[0]["observation.images.front"]
    )
    assert pickle.loads(pickle.dumps(joined))._frames == {}
