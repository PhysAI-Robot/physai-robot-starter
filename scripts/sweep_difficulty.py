"""Run `visual_servo` over difficulty levels in MuJoCo and tabulate each cell.

    uv run python scripts/sweep_difficulty.py --out outputs/difficulty
    uv run python scripts/sweep_difficulty.py --axis lighting --seed 100 --episodes 50
    uv run python scripts/sweep_difficulty.py --table-only --out outputs/difficulty

Each cell is one `eval_policy.py` run (own process, own JSON in `--out`)
with only that axis changed and friction and mass left nominal. The default
seeds (100 to 149) do not overlap the 0 to 99 seeds the policy was tuned on, so
the baseline is held out. Camera shift moves every camera and, except for the
`calibrated` level, does not tell the policy: its calibration stays at the
nominal pose, as for a bumped or mis-mounted camera. After the cells run (or
with `--table-only`) a Markdown table lists success with its Wilson interval,
place error, settling time and the most common failure category per cell.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import _bootstrap  # noqa: F401
from _cli import new_parser
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


def supported_cells(cells: list[Cell], sim: str | None) -> list[Cell]:
    """The cells `eval_policy.py` can run on `sim`: Isaac has no clutter and no
    camera shift the policy is told about."""
    if sim != "isaac":
        return cells
    return [
        cell
        for cell in cells
        if cell.axis in {"baseline", "lighting"}
        or (cell.axis == "camera" and "--camera-shift-unknown" in cell.flags)
    ]


def select_cells(axis: str | None) -> list[Cell]:
    """The baseline plus every cell of `axis` (all cells when `axis` is None)."""
    if axis is None:
        return list(CELLS)
    chosen = [cell for cell in CELLS if cell.axis in {"baseline", axis}]
    if len(chosen) == 1:
        raise ValueError(f"unknown axis {axis!r}")
    return chosen


def run_cell(
    cell: Cell,
    out_dir: Path,
    seed: int,
    episodes: int,
    max_steps: int | None,
    sim: str | None = None,
):
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
        *(["--max-steps", str(max_steps)] if max_steps is not None else []),
        *(["--sim", sim] if sim else []),
        "--nominal-physics",
        "--json",
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
    ap = new_parser(__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/difficulty"),
        help="directory for one JSON and log per cell, and table.md",
    )
    ap.add_argument(
        "--sim",
        choices=("mujoco", "isaac"),
        help="simulator engine (default: the manifest's); isaac runs only the "
        "baseline, the lighting levels and the uncalibrated camera shifts, the "
        "cells eval_policy.py supports there",
    )
    ap.add_argument(
        "--axis",
        choices=("lighting", "camera", "clutter"),
        help="run only this axis plus the baseline (default: all axes)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=100,
        help="seed of the first episode; 100 keeps the sweep off seeds 0-99, "
        "which the policies were tuned on",
    )
    ap.add_argument("--episodes", type=int, default=50, help="episodes per cell")
    ap.add_argument(
        "--max-steps",
        type=int,
        help="override the episode length (default: the manifest's)",
    )
    ap.add_argument(
        "--table-only",
        action="store_true",
        help="skip running and tabulate the JSON already in --out",
    )
    args = ap.parse_args()
    cells = select_cells(args.axis)
    cells = supported_cells(cells, args.sim)
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.table_only:
        for cell in cells:
            print(f"running {cell.name} ...", flush=True)
            run_cell(cell, args.out, args.seed, args.episodes, args.max_steps, args.sim)
    table = render_table(cells, args.out)
    (args.out / "table.md").write_text(table, encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
