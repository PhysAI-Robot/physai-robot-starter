"""Plot where each evaluation episode started and whether it succeeded.

    python scripts/plot_workspace.py outputs/mujoco.json outputs/isaac.json --out outputs/workspace.png
    python scripts/plot_workspace.py outputs/mujoco.json --label "visual servo, MuJoCo"

One column per evaluation JSON (from `eval_policy.py --json`), two rows: where the cube
started and where the target pad was. Each episode is one point on the table seen from
above (x forward, the robot just below the plot), green for success and a red cross for failure,
over the region the study samples from. Needs matplotlib (`uv sync --extra training`).
"""

from __future__ import annotations

import json
from pathlib import Path

import _bootstrap  # noqa: F401
from _cli import new_parser
from _common_args import add_out

# The sampled region of configs/manifests/so101_randomized_pick_place.yaml.
RADIUS_RANGE = (0.16, 0.255)
Y_LIMIT = 0.14
ROWS = (("cube_start", "cube start"), ("target_pos", "target pad"))


def load_results(path: Path) -> list[dict]:
    results = json.loads(path.read_text(encoding="utf-8"))["results"]
    if not results or any(key not in results[0] for key, _ in ROWS):
        raise SystemExit(
            f"{path}: results have no cube_start/target_pos; re-run eval_policy.py "
            "on a single-cube task"
        )
    return results


def plot(files: list[Path], labels: list[str], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(
        len(ROWS),
        len(files),
        figsize=(3.8 * len(files), 3.2 * len(ROWS)),
        squeeze=False,
    )
    angle = np.linspace(
        -np.arcsin(Y_LIMIT / RADIUS_RANGE[1]), np.arcsin(Y_LIMIT / RADIUS_RANGE[1]), 80
    )
    for column, (path, label) in enumerate(zip(files, labels)):
        results = load_results(path)
        wins = sum(bool(r["success"]) for r in results)
        for row, (key, title) in enumerate(ROWS):
            ax = axes[row][column]
            for radius in RADIUS_RANGE:
                t = angle[np.abs(radius * np.sin(angle)) <= Y_LIMIT]
                ax.plot(radius * np.sin(t), radius * np.cos(t), color="0.75", lw=1)
            for ok, style in (
                (True, dict(c="tab:green", marker="o", s=14)),
                (False, dict(c="tab:red", marker="x", s=46, linewidths=1.8)),
            ):
                points = np.array([r[key] for r in results if bool(r["success"]) is ok])
                if len(points):
                    ax.scatter(
                        points[:, 1], points[:, 0], zorder=3 if not ok else 2, **style
                    )
            ax.set_xlim(
                Y_LIMIT + 0.03, -Y_LIMIT - 0.03
            )  # +y on the left, as the front camera sees it
            ax.set_ylim(0.12, RADIUS_RANGE[1] + 0.02)
            ax.set_aspect("equal")
            ax.set_xlabel("y (m)")
            ax.set_ylabel("x (m), robot at 0")
            ax.set_title(f"{label}: {wins}/{len(results)}\n{title}", fontsize=10)
    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=160)
    print(f"workspace plot -> {out}")


def main() -> int:
    ap = new_parser(__doc__)
    ap.add_argument("results", type=Path, nargs="+", help="evaluation JSON files")
    ap.add_argument(
        "--label",
        action="append",
        help="column title, once per file (default: file name)",
    )
    add_out(ap, default=Path("outputs/workspace.png"), help="image to write")
    args = ap.parse_args()
    labels = args.label or [path.stem for path in args.results]
    if len(labels) != len(args.results):
        ap.error("give one --label per results file, or none")
    plot(args.results, labels, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
