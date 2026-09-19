import json
import sys
from pathlib import Path

import pytest

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
