import json
import sys
from pathlib import Path

import pytest

from physai.data.evaluation import EvaluationReport

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def evaluation(*, failed_seeds=(), collisions=0, timeouts=0, unsafe=0) -> dict:
    results = [
        {
            "seed": seed,
            "success": seed not in failed_seeds,
            "steps": 600 if seed in failed_seeds else 170,
            "failure_reason": "timeout" if seed in failed_seeds else None,
        }
        for seed in range(4)
    ]
    successes = sum(result["success"] for result in results)
    return {
        "schema_version": "physai.evaluation.v1",
        "policy": "visual_servo",
        "robot": "so101",
        "task": "pick_place",
        "summary": {
            "episodes": 4,
            "success_count": successes,
            "success_rate": successes / 4,
            "collision_count": collisions,
            "timeout_count": timeouts,
            "unsafe_action_count": unsafe,
            "mean_reward": 12.5,
            "mean_steps": 220.0,
        },
        "results": results,
    }


def run_report(monkeypatch, capsys, tmp_path, data: dict, *flags: str):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import report_evaluation

    path = tmp_path / "eval.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["report_evaluation.py", str(path), *flags])
    code = report_evaluation.main()
    return code, capsys.readouterr().out


def test_clean_run_passes_the_gate_and_reports_the_numbers(
    monkeypatch, capsys, tmp_path
):
    code, out = run_report(
        monkeypatch, capsys, tmp_path, evaluation(), "--require-all-success"
    )

    assert code == 0
    assert "| 4 | 4/4 (100%) | 0 | 0 | 0 | 220.0 | 12.50 |" in out
    assert "Seeds 0 to 3." in out
    assert "Every episode succeeded" in out


def test_failed_episode_fails_the_gate_and_is_still_reported(
    monkeypatch, capsys, tmp_path
):
    data = evaluation(failed_seeds=(2,), timeouts=1)

    code, out = run_report(monkeypatch, capsys, tmp_path, data, "--require-all-success")

    assert code == 1
    assert "3/4 (75%)" in out
    assert "| 2 | 600 | timeout |" in out
    assert "1 of 4 episodes failed" in out
    assert "1 timeout event(s)" in out


def test_safety_events_fail_the_gate_even_when_every_episode_succeeded(
    monkeypatch, capsys, tmp_path
):
    data = evaluation(collisions=2, unsafe=1)

    code, out = run_report(monkeypatch, capsys, tmp_path, data, "--require-all-success")

    assert code == 1
    assert "2 collision event(s); 1 unsafe action event(s)" in out


def test_without_the_gate_flag_a_failing_run_still_exits_zero(
    monkeypatch, capsys, tmp_path
):
    code, out = run_report(monkeypatch, capsys, tmp_path, evaluation(failed_seeds=(0,)))

    assert code == 0
    assert "Gate failed" in out


def test_unknown_schema_version_is_rejected(monkeypatch, capsys, tmp_path):
    data = evaluation() | {"schema_version": "something.else"}

    with pytest.raises(SystemExit) as exit_info:
        run_report(monkeypatch, capsys, tmp_path, data)

    assert exit_info.value.code == 2
    assert "expected 'physai.evaluation.v1'" in capsys.readouterr().err


def shard(
    first_seed: int, count: int, *, failed=(), collided=(), policy="visual_servo"
):
    """One CI shard, built with the same report class eval_policy.py uses."""
    results = tuple(
        {
            "seed": seed,
            "success": seed not in failed,
            "steps": 170,
            "reward": 12.0,
            "timeout": False,
            "collision": seed in collided,
            "unsafe_action": False,
            "failure_reason": "stuck" if seed in failed else None,
        }
        for seed in range(first_seed, first_seed + count)
    )
    return EvaluationReport(policy, "so101", "pick_place", results).to_dict()


def run_merge(monkeypatch, capsys, tmp_path, shards: list[dict], *flags: str):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import report_evaluation

    paths = []
    for index, data in enumerate(shards):
        path = tmp_path / f"shard-{index}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        paths.append(str(path))
    monkeypatch.setattr(sys, "argv", ["report_evaluation.py", *paths, *flags])
    code = report_evaluation.main()
    return code, capsys.readouterr().out


def test_shards_merge_into_one_report_regardless_of_file_order(
    monkeypatch, capsys, tmp_path
):
    shards = [shard(4, 4), shard(0, 4)]

    code, out = run_merge(
        monkeypatch,
        capsys,
        tmp_path,
        shards,
        "--require-all-success",
        "--expect-episodes",
        "8",
    )

    assert code == 0
    assert "| 8 | 8/8 (100%) |" in out
    assert "Seeds 0 to 7." in out


def test_merged_summary_counts_safety_events_from_every_shard(
    monkeypatch, capsys, tmp_path
):
    shards = [shard(0, 4, collided=(1,)), shard(4, 4, failed=(6,), collided=(7,))]

    code, out = run_merge(
        monkeypatch, capsys, tmp_path, shards, "--require-all-success"
    )

    assert code == 1
    assert "7/8 (88%)" in out
    assert "| 6 | 170 | stuck |" in out
    assert "2 collision event(s)" in out


def test_a_missing_shard_fails_the_gate_even_though_the_rest_succeeded(
    monkeypatch, capsys, tmp_path
):
    shards = [shard(0, 4), shard(4, 4), shard(8, 4)]

    code, out = run_merge(
        monkeypatch,
        capsys,
        tmp_path,
        shards,
        "--require-all-success",
        "--expect-episodes",
        "16",
    )

    assert code == 1
    assert "expected 16 episodes, found 12" in out
    assert "Every episode succeeded" not in out


def test_the_episode_count_alone_does_not_gate_without_the_success_flag(
    monkeypatch, capsys, tmp_path
):
    code, out = run_merge(
        monkeypatch, capsys, tmp_path, [shard(0, 4)], "--expect-episodes", "8"
    )

    assert code == 0
    assert "expected 8 episodes, found 4" in out


def test_shards_that_share_a_seed_are_rejected(monkeypatch, capsys, tmp_path):
    with pytest.raises(SystemExit) as exit_info:
        run_merge(monkeypatch, capsys, tmp_path, [shard(0, 4), shard(3, 4)])

    assert exit_info.value.code == 2
    assert "more than one file: [3]" in capsys.readouterr().err


def test_shards_from_different_policies_are_rejected(monkeypatch, capsys, tmp_path):
    with pytest.raises(SystemExit) as exit_info:
        run_merge(
            monkeypatch,
            capsys,
            tmp_path,
            [shard(0, 2), shard(2, 2, policy="scripted")],
        )

    assert exit_info.value.code == 2
    assert "cannot merge" in capsys.readouterr().err


def test_merged_evaluation_can_be_written_out(monkeypatch, capsys, tmp_path):
    out_path = tmp_path / "merged" / "eval.json"

    code, _ = run_merge(
        monkeypatch,
        capsys,
        tmp_path,
        [shard(0, 3), shard(3, 3)],
        "--merged-out",
        str(out_path),
    )

    merged = json.loads(out_path.read_text(encoding="utf-8"))
    assert code == 0
    assert merged["summary"]["episodes"] == 6
    assert [result["seed"] for result in merged["results"]] == list(range(6))
