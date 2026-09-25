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

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from physai.data.evaluation import EvaluationReport

SCHEMA_VERSION = "physai.evaluation.v1"
COUNTERS = {
    "collision_count": "collision",
    "timeout_count": "timeout",
    "unsafe_action_count": "unsafe action",
}


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
    ]
    failed = [result for result in data["results"] if not result["success"]]
    if failed:
        lines += ["", "| Failed seed | Steps | Reason |", "| ---: | ---: | --- |"]
        lines += [
            f"| {r['seed']} | {r['steps']} | {r.get('failure_reason') or 'unknown'} |"
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
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "results",
        type=Path,
        nargs="+",
        help="JSON from eval_policy.py --json-out; several files are merged",
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
        "--merged-out",
        type=Path,
        help="write the merged evaluation JSON here",
    )
    args = parser.parse_args()

    try:
        parts = [load_evaluation(path) for path in args.results]
        data = parts[0] if len(parts) == 1 else merge_evaluations(parts)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    if args.merged_out:
        args.merged_out.parent.mkdir(parents=True, exist_ok=True)
        args.merged_out.write_text(json.dumps(data, indent=2), encoding="utf-8")

    failures = gate_failures(data["summary"], args.expect_episodes)
    sys.stdout.write(render(data, failures))
    return 1 if args.require_all_success and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
