import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import eval_policy_sharded  # noqa: E402


def test_shards_cover_every_seed_once_and_the_last_may_be_short():
    shards = eval_policy_sharded.shard_ranges(seed=5, episodes=25, shard_size=10)
    assert shards == [(5, 10), (15, 10), (25, 5)]
    covered = [s for start, count in shards for s in range(start, start + count)]
    assert covered == list(range(5, 30))


@pytest.mark.parametrize("episodes, size", [(0, 10), (10, 0)])
def test_shards_reject_empty_input(episodes, size):
    with pytest.raises(ValueError):
        eval_policy_sharded.shard_ranges(0, episodes, size)


def test_a_byte_the_console_code_page_cannot_decode_does_not_crash_the_wrapper(
    tmp_path,
):
    script = tmp_path / "child.py"
    script.write_text(
        r"import sys; sys.stdout.buffer.write(b'ok \x8d end')", encoding="utf-8"
    )
    out = tmp_path / "shard-0.json"
    out.write_text("{}", encoding="utf-8")  # the child ignores the arguments
    code, glitch, _ = eval_policy_sharded.run_shard(0, 1, out, [], script)
    assert (code, glitch) == (0, False)
    assert "ok" in out.with_suffix(".log").read_text(encoding="utf-8")
