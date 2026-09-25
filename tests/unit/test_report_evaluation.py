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


def test_a_single_evaluation_is_reported_and_gated(monkeypatch, capsys, tmp_path):
    gate = "--require-all-success"

    code, out = run_report(monkeypatch, capsys, tmp_path, evaluation(), gate)
    assert code == 0
    assert "| 4 | 4/4 (100%) | 0 | 0 | 0 | 220.0 | 12.50 |" in out
    assert "Seeds 0 to 3." in out
    assert "Every episode succeeded" in out

    failed = evaluation(failed_seeds=(2,), timeouts=1)
    code, out = run_report(monkeypatch, capsys, tmp_path, failed, gate)
    assert code == 1
    assert "3/4 (75%)" in out
    assert "| 2 | 600 | timeout |" in out
    assert "1 of 4 episodes failed" in out
    assert "1 timeout event(s)" in out

    # safety events fail the gate even when every episode succeeded
    unsafe = evaluation(collisions=2, unsafe=1)
    code, out = run_report(monkeypatch, capsys, tmp_path, unsafe, gate)
    assert code == 1
    assert "2 collision event(s); 1 unsafe action event(s)" in out

    # without the flag a failing run is reported but still exits zero
    code, out = run_report(monkeypatch, capsys, tmp_path, evaluation(failed_seeds=(0,)))
    assert code == 0
    assert "Gate failed" in out

    with pytest.raises(SystemExit) as exit_info:
        run_report(
            monkeypatch, capsys, tmp_path, evaluation() | {"schema_version": "other"}
        )
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


def test_shards_merge_into_one_report_and_can_be_written_out(
    monkeypatch, capsys, tmp_path
):
    gate = ("--require-all-success", "--expect-episodes", "8")
    # file order does not matter
    code, out = run_merge(
        monkeypatch, capsys, tmp_path, [shard(4, 4), shard(0, 4)], *gate
    )
    assert code == 0
    assert "| 8 | 8/8 (100%) |" in out
    assert "Seeds 0 to 7." in out

    # safety events from every shard are counted
    mixed = [shard(0, 4, collided=(1,)), shard(4, 4, failed=(6,), collided=(7,))]
    code, out = run_merge(monkeypatch, capsys, tmp_path, mixed, "--require-all-success")
    assert code == 1
    assert "7/8 (88%)" in out
    assert "| 6 | 170 | stuck |" in out
    assert "2 collision event(s)" in out

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


def test_a_missing_shard_fails_the_gate_only_when_success_is_required(
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

    # the episode count alone does not gate without the success flag
    code, out = run_merge(
        monkeypatch, capsys, tmp_path, [shard(0, 4)], "--expect-episodes", "8"
    )
    assert code == 0
    assert "expected 8 episodes, found 4" in out


def test_shards_that_overlap_or_disagree_are_rejected(monkeypatch, capsys, tmp_path):
    with pytest.raises(SystemExit) as exit_info:
        run_merge(monkeypatch, capsys, tmp_path, [shard(0, 4), shard(3, 4)])
    assert exit_info.value.code == 2
    assert "more than one file: [3]" in capsys.readouterr().err

    with pytest.raises(SystemExit) as exit_info:
        run_merge(
            monkeypatch,
            capsys,
            tmp_path,
            [shard(0, 2), shard(2, 2, policy="scripted")],
        )
    assert exit_info.value.code == 2
    assert "cannot merge" in capsys.readouterr().err
