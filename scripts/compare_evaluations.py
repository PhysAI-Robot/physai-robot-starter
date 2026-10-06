"""Compare two evaluations of the same policy, e.g. MuJoCo against Isaac Sim.

    uv run python scripts/compare_evaluations.py outputs/mujoco.json outputs/isaac.json
    uv run python scripts/compare_evaluations.py a.json b.json --label-a MuJoCo --label-b Isaac

Both files come from `eval_policy.py --json-out` over the same seeds (for
`--sim isaac` a seed places the cube where MuJoCo does). The Markdown reports
each side's success rate with a Wilson 95% interval, how the seeds agree
between the two (both succeed, only one does, neither), the mean steps, and
the seeds on which they differ, so a gap is traceable to specific layouts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from _cli import new_parser
from report_evaluation import load_evaluation, wilson_interval  # noqa: F401


def seed_agreement(a: list[dict], b: list[dict]) -> dict:
    """How the two evaluations' per-seed outcomes line up on common seeds."""
    by_seed_a = {r["seed"]: bool(r["success"]) for r in a}
    by_seed_b = {r["seed"]: bool(r["success"]) for r in b}
    common = sorted(set(by_seed_a) & set(by_seed_b))
    result = {"both": [], "only_a": [], "only_b": [], "neither": []}
    for seed in common:
        key = {
            (True, True): "both",
            (True, False): "only_a",
            (False, True): "only_b",
            (False, False): "neither",
        }[(by_seed_a[seed], by_seed_b[seed])]
        result[key].append(seed)
    result["only_in_a"] = sorted(set(by_seed_a) - set(by_seed_b))
    result["only_in_b"] = sorted(set(by_seed_b) - set(by_seed_a))
    return result


def _format_seeds(seeds: list[int], limit: int = 12) -> str:
    if not seeds:
        return "-"
    shown = ", ".join(str(seed) for seed in seeds[:limit])
    return shown + (f", ... ({len(seeds)} total)" if len(seeds) > limit else "")


def render(a: dict, b: dict, label_a: str, label_b: str) -> str:
    lines = [
        f"### {a['robot']} {a['policy']} ({a['task']}): {label_a} vs {label_b}",
        "",
    ]
    lines += [
        "| Simulator | Episodes | Success | Wilson 95% | Mean steps |",
        "| --- | ---: | ---: | --- | ---: |",
    ]
    for label, data in ((label_a, a), (label_b, b)):
        summary = data["summary"]
        low, high = wilson_interval(summary["success_count"], summary["episodes"])
        lines.append(
            f"| {label} | {summary['episodes']} "
            f"| {summary['success_count']} ({summary['success_rate']:.0%}) "
            f"| {low:.0%} to {high:.0%} | {summary['mean_steps']:.0f} |"
        )
    agreement = seed_agreement(a["results"], b["results"])
    common = sum(len(agreement[key]) for key in ("both", "only_a", "only_b", "neither"))
    lines += [
        "",
        f"Per-seed agreement over {common} common seeds:",
        "",
        "| Outcome | Seeds | Which |",
        "| --- | ---: | --- |",
        f"| both succeed | {len(agreement['both'])} | |",
        f"| only {label_a} succeeds | {len(agreement['only_a'])} "
        f"| {_format_seeds(agreement['only_a'])} |",
        f"| only {label_b} succeeds | {len(agreement['only_b'])} "
        f"| {_format_seeds(agreement['only_b'])} |",
        f"| neither succeeds | {len(agreement['neither'])} "
        f"| {_format_seeds(agreement['neither'])} |",
    ]
    for key, label in (("only_in_a", label_a), ("only_in_b", label_b)):
        if agreement[key]:
            lines += ["", f"Seeds only in {label}: {_format_seeds(agreement[key])}."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = new_parser(__doc__)
    parser.add_argument("a", type=Path, help="first evaluation JSON")
    parser.add_argument("b", type=Path, help="second evaluation JSON")
    parser.add_argument(
        "--label-a", default="MuJoCo", help="name of the first evaluation in the report"
    )
    parser.add_argument(
        "--label-b", default="Isaac", help="name of the second evaluation in the report"
    )
    args = parser.parse_args()
    try:
        a, b = load_evaluation(args.a), load_evaluation(args.b)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    identity = (a["policy"], a["robot"], a["task"])
    if identity != (b["policy"], b["robot"], b["task"]):
        parser.error(f"evaluations differ in policy/robot/task: {identity}")
    sys.stdout.write(render(a, b, args.label_a, args.label_b))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
