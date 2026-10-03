import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import sweep_difficulty  # noqa: E402


def evaluation(outcomes: dict[int, str | None]) -> dict:
    """`outcomes` maps seed to None (success) or the phase it timed out in."""
    results = [
        {
            "seed": seed,
            "success": phase is None,
            "timeout": phase is not None,
            "phase": phase or "RELEASE",
            "steps": 600 if phase else 200,
            "dist_cube_target": 0.006 if phase is None else 0.2,
            "settling_time_s": 1.0,
        }
        for seed, phase in outcomes.items()
    ]
    wins = sum(r["success"] for r in results)
    return {
        "schema_version": "physai.evaluation.v1",
        "policy": "visual_servo",
        "robot": "so101",
        "task": "single_cube_fixed_place",
        "summary": {
            "episodes": len(results),
            "success_count": wins,
            "success_rate": wins / len(results),
        },
        "results": results,
    }


def test_cell_names_are_unique_and_start_with_the_baseline():
    names = [cell.name for cell in sweep_difficulty.CELLS]
    assert len(names) == len(set(names))
    assert sweep_difficulty.CELLS[0].axis == "baseline"


def test_a_camera_shift_is_unknown_to_the_policy_unless_marked_calibrated():
    cells = {
        cell.level: cell for cell in sweep_difficulty.CELLS if cell.axis == "camera"
    }
    assert "--camera-shift-unknown" in cells["10mm"].flags
    assert cells["10mm"].flags[:2] == ("--camera-jitter", "0.01")
    assert "--camera-shift-unknown" not in cells["20mm_calibrated"].flags


def test_select_cells_keeps_the_baseline_and_rejects_unknown_axes():
    chosen = sweep_difficulty.select_cells("clutter")
    assert {cell.axis for cell in chosen} == {"baseline", "clutter"}
    with pytest.raises(ValueError):
        sweep_difficulty.select_cells("fog")


def test_table_reports_each_cell_and_marks_missing_ones(tmp_path):
    baseline, lighting = sweep_difficulty.select_cells("lighting")[:2]
    (tmp_path / f"{baseline.name}.json").write_text(
        json.dumps(evaluation({0: None, 1: None, 2: "CLOSE", 3: "CLOSE"})),
        encoding="utf-8",
    )
    table = sweep_difficulty.render_table([baseline, lighting], tmp_path)
    assert "| baseline | nominal | 2/4 |" in table
    assert "timeout_in_close x2" in table
    assert "| 6.0 / 6.0 |" in table
    assert f"| lighting | {lighting.level} | not run |" in table
