import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import compare_evaluations  # noqa: E402


def evaluation(outcomes: dict[int, bool]) -> dict:
    results = [
        {"seed": seed, "success": ok, "steps": 150 if ok else 600}
        for seed, ok in outcomes.items()
    ]
    successes = sum(outcomes.values())
    return {
        "policy": "visual_servo",
        "robot": "so101",
        "task": "pick_place",
        "summary": {
            "episodes": len(results),
            "success_count": successes,
            "success_rate": successes / len(results),
            "mean_steps": sum(r["steps"] for r in results) / len(results),
        },
        "results": results,
    }


def test_wilson_interval_matches_known_values():
    low, high = compare_evaluations.wilson_interval(95, 100)
    assert (low, high) == pytest.approx((0.8884, 0.9784), abs=5e-4)
    assert compare_evaluations.wilson_interval(0, 10)[0] == pytest.approx(0.0)
    assert compare_evaluations.wilson_interval(10, 10)[1] == pytest.approx(1.0)
    # 100/100 is not "certainly 100%": the lower bound is the rule of three-ish.
    assert compare_evaluations.wilson_interval(100, 100)[0] == pytest.approx(
        0.963, abs=2e-3
    )
    with pytest.raises(ValueError):
        compare_evaluations.wilson_interval(1, 0)
    with pytest.raises(ValueError):
        compare_evaluations.wilson_interval(5, 4)


def test_seed_agreement_buckets_each_common_seed():
    a = evaluation({0: True, 1: True, 2: False, 3: False, 4: True})
    b = evaluation({0: True, 1: False, 2: True, 3: False, 9: True})

    agreement = compare_evaluations.seed_agreement(a["results"], b["results"])

    assert agreement["both"] == [0]
    assert agreement["only_a"] == [1]
    assert agreement["only_b"] == [2]
    assert agreement["neither"] == [3]
    assert agreement["only_in_a"] == [4] and agreement["only_in_b"] == [9]


def test_render_names_the_simulators_and_the_seeds_that_differ():
    a = evaluation({0: True, 1: True, 2: True})
    b = evaluation({0: True, 1: False, 2: True})

    text = compare_evaluations.render(a, b, "MuJoCo", "Isaac")

    assert "MuJoCo vs Isaac" in text
    assert "| MuJoCo | 3 | 3 (100%)" in text
    assert "| Isaac | 3 | 2 (67%)" in text
    assert "only MuJoCo succeeds | 1 | 1 |" in text
