"""Summarize an evaluation JSON as Markdown, optionally failing on any failure.

    uv run python scripts/report_evaluation.py outputs/eval.json
    uv run python scripts/report_evaluation.py outputs/eval.json --require-all-success

The Markdown goes to stdout, so CI can append it to the job summary. With
``--require-all-success`` the exit code is 1 when any episode failed or the
collision, timeout, or unsafe-action counters are not zero. The Markdown is
still printed in that case, so a failing run shows its numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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


def gate_failures(summary: dict) -> list[str]:
    failures = []
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
        f"Seeds {min(seeds)} to {max(seeds)}.",
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
        "results", type=Path, help="JSON from eval_policy.py --json-out"
    )
    parser.add_argument(
        "--require-all-success",
        action="store_true",
        help="exit 1 unless every episode succeeded with zero safety events",
    )
    args = parser.parse_args()

    try:
        data = load_evaluation(args.results)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    failures = gate_failures(data["summary"])
    sys.stdout.write(render(data, failures))
    return 1 if args.require_all_success and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
