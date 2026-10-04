import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_sharded_eval  # noqa: E402


def test_shards_cover_every_seed_once_and_the_last_may_be_short():
    shards = run_sharded_eval.shard_ranges(seed=5, episodes=25, shard_size=10)
    assert shards == [(5, 10), (15, 10), (25, 5)]
    covered = [s for start, count in shards for s in range(start, start + count)]
    assert covered == list(range(5, 30))


@pytest.mark.parametrize("episodes, size", [(0, 10), (10, 0)])
def test_shards_reject_empty_input(episodes, size):
    with pytest.raises(ValueError):
        run_sharded_eval.shard_ranges(0, episodes, size)
