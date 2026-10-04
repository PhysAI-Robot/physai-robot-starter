"""Run `visual_servo` over difficulty levels in MuJoCo and tabulate each cell.

    uv run python scripts/sweep_difficulty.py --out-dir outputs/difficulty
    uv run python scripts/sweep_difficulty.py --axis lighting --seed 100 --episodes 50
    uv run python scripts/sweep_difficulty.py --table-only --out-dir outputs/difficulty

Each cell is one `eval_policy.py` run (own process, own JSON in `--out-dir`)
with only that axis changed and friction and mass left nominal. The default
seeds (100 to 149) do not overlap the 0 to 99 seeds the policy was tuned on, so
the baseline is held out. Camera shift moves every camera and, except for the
`calibrated` level, does not tell the policy: its calibration stays at the
nominal pose, as for a bumped or mis-mounted camera. After the cells run (or
with `--table-only`) a Markdown table lists success with its Wilson interval,
place error, settling time and the most common failure category per cell.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import _bootstrap  # noqa: F401
from report_evaluation import (
    failure_histogram,
    load_evaluation,
    metric_rows,
    wilson_interval,
)

EVAL_SCRIPT = Path(__file__).resolve().parent / "eval_policy.py"


@dataclass(frozen=True)
class Cell:
    axis: str
    level: str
    flags: tuple[str, ...]

    @property
    def name(self) -> str:
        return f"{self.axis}_{self.level}"


def _lighting(scale: float) -> Cell:
    return Cell("lighting", f"x{scale:g}", ("--lighting-scale", str(scale)))


def _shift(millimetres: int, *, calibrated: bool = False) -> Cell:
    flags = ("--camera-jitter", str(millimetres / 1000.0))
    if calibrated:
        return Cell("camera", f"{millimetres}mm_calibrated", flags)
    return Cell("camera", f"{millimetres}mm", (*flags, "--camera-shift-unknown"))


def _clutter(count: int) -> Cell:
    return Cell("clutter", f"{count}_boxes", ("--clutter-count", str(count)))


CELLS: tuple[Cell, ...] = (
    Cell("baseline", "nominal", ()),
    *(_lighting(scale) for scale in (0.7, 0.5, 1.3, 1.6)),
    *(_shift(mm) for mm in (5, 10, 20)),
    _shift(20, calibrated=True),
    *(_clutter(count) for count in (1, 2, 4)),
)


def select_cells(axis: str | None) -> list[Cell]:
    """The baseline plus every cell of `axis` (all cells when `axis` is None)."""
    if axis is None:
        return list(CELLS)
    chosen = [cell for cell in CELLS if cell.axis in {"baseline", axis}]
    if len(chosen) == 1:
        raise ValueError(f"unknown axis {axis!r}")
    return chosen


def run_cell(cell: Cell, out_dir: Path, seed: int, episodes: int, max_steps: int):
    out = out_dir / f"{cell.name}.json"
    command = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--policy",
        "visual_servo",
        "--seed",
        str(seed),
        "--episodes",
        str(episodes),
        "--max-steps",
        str(max_steps),
        "--nominal-physics",
        "--json-out",
        str(out),
        *cell.flags,
    ]
    completed = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    out.with_suffix(".log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise SystemExit(f"{cell.name} failed; see {out.with_suffix('.log')}")


def cell_row(cell: Cell, data: dict) -> str:
    summary = data["summary"]
    low, high = wilson_interval(summary["success_count"], summary["episodes"])
    metrics = dict(metric_rows(data["results"]))
    place = metrics.get("Place error, successes (mm)", "-")
    settle = metrics.get("Settling time (s)", "-")
    histogram = failure_histogram(data["results"])
    top = ", ".join(f"{name} x{count}" for name, count in histogram.items())
    return (
        f"| {cell.axis} | {cell.level} "
        f"| {summary['success_count']}/{summary['episodes']} "
        f"| {low:.0%} to {high:.0%} | {place} | {settle} | {top or '-'} |"
    )


def render_table(cells: list[Cell], out_dir: Path) -> str:
    lines = [
        "| Axis | Level | Success | Wilson 95% | Place error mm (median / p90) "
        "| Settling s (median / p90) | Failures |",
        "| --- | --- | ---: | --- | ---: | ---: | --- |",
    ]
    for cell in cells:
        path = out_dir / f"{cell.name}.json"
        if path.exists():
            lines.append(cell_row(cell, load_evaluation(path)))
        else:
            lines.append(f"| {cell.axis} | {cell.level} | not run | | | | |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/difficulty"))
    ap.add_argument("--axis", choices=("lighting", "camera", "clutter"))
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument(
        "--table-only",
        action="store_true",
        help="skip running and tabulate the JSON already in --out-dir",
    )
    args = ap.parse_args()
    cells = select_cells(args.axis)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.table_only:
        for cell in cells:
            print(f"running {cell.name} ...", flush=True)
            run_cell(cell, args.out_dir, args.seed, args.episodes, args.max_steps)
    table = render_table(cells, args.out_dir)
    (args.out_dir / "table.md").write_text(table, encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
