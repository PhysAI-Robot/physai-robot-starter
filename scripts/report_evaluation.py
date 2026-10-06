"""Summarize evaluation JSON as Markdown, optionally failing on any failure.

    uv run python scripts/report_evaluation.py outputs/eval.json
    uv run python scripts/report_evaluation.py outputs/eval.json --require-all-success
    uv run python scripts/report_evaluation.py shard-*.json --expect-episodes 20 \\
        --require-all-success

Several files are merged into one report, which is how an evaluation split
across parallel CI jobs is put back together. They must come from the same
policy, robot, and task and must not share a seed.

The Markdown goes to stdout, so CI can append it to the job summary. With
``--require-all-success`` the exit code is 1 when any episode failed or the
collision, timeout, or unsafe-action counters are not zero. ``--expect-episodes``
adds the check that exactly that many episodes are present to that gate, so a
shard that never reported cannot pass as a smaller successful run. The Markdown is still
printed when the gate fails, so a failing run shows its numbers.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from _cli import new_parser
import numpy as np
from physai.data.evaluation import EvaluationReport

SCHEMA_VERSION = "physai.evaluation.v1"
COUNTERS = {
    "collision_count": "collision",
    "timeout_count": "timeout",
    "unsafe_action_count": "unsafe action",
}


Z_95 = 1.959963984540054


def wilson_interval(successes: int, total: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default)."""
    if total <= 0:
        raise ValueError("total must be positive")
    if not 0 <= successes <= total:
        raise ValueError("successes must be between 0 and total")
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    margin /= denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def failure_category(result: dict) -> str | None:
    """Why an episode failed, from fields every evaluation already records.

    `None` for a success. A perception failure carries the policy's own
    `failure_reason`; a timeout is named after the phase the policy was stuck
    in, which is the useful part (a stall in APPROACH is a perception or
    reachability problem, one in CLOSE is a grasp problem).
    """
    if result["success"]:
        return None
    if result.get("unsafe_action"):
        return "unsafe_action"
    if result.get("collision"):
        return "collision"
    reason = result.get("failure_reason")
    if reason:
        if reason == "grasp_missed":
            return reason
        if reason.startswith("missing_camera") or reason == "feature_not_found":
            return reason.split(":")[0]
        return "policy_error"
    phase = result.get("phase")
    if phase == "DONE":
        # The state machine ran every phase and stopped, but the cube is not on
        # the target: a grasp or transfer that went wrong without being noticed.
        return "finished_not_placed"
    if result.get("timeout") and phase:
        return f"timeout_in_{phase.lower()}"
    return "timeout" if result.get("timeout") else "unknown"


def _percentiles(values: list[float]) -> tuple[float, float] | None:
    if not values:
        return None
    return float(np.percentile(values, 50)), float(np.percentile(values, 90))


def metric_rows(results: list[dict]) -> list[tuple[str, str]]:
    """Position error and settling time as `median / p90` strings, when recorded.

    Place error is the cube-to-target distance at the end of a successful
    episode; settling time is how long the pinch took to first reach its
    first waypoint (`settling_time_s`, recorded by `visual_servo`).
    """
    rows = []
    place = _percentiles(
        [
            1000.0 * r["dist_cube_target"]
            for r in results
            if r["success"] and r.get("dist_cube_target") is not None
        ]
    )
    if place:
        rows.append(("Place error, successes (mm)", f"{place[0]:.1f} / {place[1]:.1f}"))
    settle = _percentiles(
        [r["settling_time_s"] for r in results if r.get("settling_time_s") is not None]
    )
    if settle:
        rows.append(("Settling time (s)", f"{settle[0]:.2f} / {settle[1]:.2f}"))
    return rows


def failure_histogram(results: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        category = failure_category(result)
        if category:
            counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def load_evaluation(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path} has schema_version {data.get('schema_version')!r}, "
            f"expected {SCHEMA_VERSION!r}"
        )
    return data


def merge_evaluations(parts: list[dict]) -> dict:
    """Combine evaluations of the same policy into one, recomputing the summary."""
    first = parts[0]
    identity = (first["policy"], first["robot"], first["task"])
    for part in parts[1:]:
        if (part["policy"], part["robot"], part["task"]) != identity:
            raise ValueError(
                f"cannot merge {identity} with "
                f"{(part['policy'], part['robot'], part['task'])}"
            )
    results = [result for part in parts for result in part["results"]]
    seeds = [result["seed"] for result in results]
    duplicated = sorted({seed for seed in seeds if seeds.count(seed) > 1})
    if duplicated:
        raise ValueError(f"seeds appear in more than one file: {duplicated}")
    results.sort(key=lambda result: result["seed"])
    return EvaluationReport(*identity, results=tuple(results)).to_dict()


def gate_failures(summary: dict, expected_episodes: int | None = None) -> list[str]:
    failures = []
    if expected_episodes is not None and summary["episodes"] != expected_episodes:
        failures.append(
            f"expected {expected_episodes} episodes, found {summary['episodes']}"
        )
    failed = summary["episodes"] - summary["success_count"]
    if failed:
        failures.append(f"{failed} of {summary['episodes']} episodes failed")
    for key, label in COUNTERS.items():
        if summary[key]:
            failures.append(f"{summary[key]} {label} event(s)")
    return failures


def render(data: dict, failures: list[str]) -> str:
    summary = data["summary"]
    episodes = summary["episodes"]
    seeds = [result["seed"] for result in data["results"]]
    low, high = wilson_interval(summary["success_count"], episodes)
    lines = [
        f"### {data['robot']} {data['policy']} evaluation ({data['task']})",
        "",
        "| Episodes | Success | Collisions | Timeouts | Unsafe actions "
        "| Mean steps | Mean reward |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {episodes} "
        f"| {summary['success_count']}/{episodes} ({summary['success_rate']:.0%}) "
        f"| {summary['collision_count']} "
        f"| {summary['timeout_count']} "
        f"| {summary['unsafe_action_count']} "
        f"| {summary['mean_steps']:.1f} "
        f"| {summary['mean_reward']:.2f} |",
        "",
        f"Seeds {min(seeds)} to {max(seeds)}." if seeds else "No episodes.",
        "",
        f"Success {summary['success_rate']:.0%}, Wilson 95% interval "
        f"{low:.0%} to {high:.0%}.",
    ]
    rows = metric_rows(data["results"])
    if rows:
        lines += ["", "| Metric (median / p90) | Value |", "| --- | ---: |"]
        lines += [f"| {name} | {value} |" for name, value in rows]
    histogram = failure_histogram(data["results"])
    if histogram:
        lines += ["", "| Failure category | Episodes |", "| --- | ---: |"]
        lines += [f"| {name} | {count} |" for name, count in histogram.items()]
    failed = [result for result in data["results"] if not result["success"]]
    if failed:
        lines += ["", "| Failed seed | Steps | Reason |", "| ---: | ---: | --- |"]
        lines += [
            f"| {r['seed']} | {r['steps']} | {r.get('failure_reason') or failure_category(r) or 'unknown'} |"
            for r in failed
        ]
    lines += [""]
    lines += (
        [f"**Gate failed:** {'; '.join(failures)}."]
        if failures
        else [
            "Every episode succeeded with no collisions, timeouts, or unsafe actions."
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = new_parser(__doc__)
    parser.add_argument(
        "results",
        type=Path,
        nargs="+",
        help="JSON from eval_policy.py --json; several files are merged",
    )
    parser.add_argument(
        "--require-all-success",
        action="store_true",
        help="exit 1 unless every episode succeeded with zero safety events",
    )
    parser.add_argument(
        "--expect-episodes",
        type=int,
        help="add a gate check that exactly this many episodes are present",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="write the merged evaluation JSON here",
    )
    args = parser.parse_args()

    try:
        parts = [load_evaluation(path) for path in args.results]
        data = parts[0] if len(parts) == 1 else merge_evaluations(parts)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(data, indent=2), encoding="utf-8")

    failures = gate_failures(data["summary"], args.expect_episodes)
    sys.stdout.write(render(data, failures))
    return 1 if args.require_all_success and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
