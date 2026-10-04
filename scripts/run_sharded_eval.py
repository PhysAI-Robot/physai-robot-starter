"""Run `eval_policy.py` in small shards, each in its own process, and merge them.

    OMNI_KIT_ACCEPT_EULA=YES uv run python scripts/run_sharded_eval.py \\
        --out-dir outputs/eval/isaac_rerun --seed 0 --episodes 100 --shard-size 10 \\
        -- --sim isaac --policy visual_servo --max-steps 600

Everything after `--` goes to `eval_policy.py`. Isaac Sim sometimes stops
drawing the robot for a whole process (see `research/classical_control/FINDINGS.md`),
and `eval_policy.py` then exits without a result; a shard that fails is retried
up to `--retries` times in a fresh process. Every attempt is appended to
`attempts.jsonl` (start time, duration, exit code, whether it was the render
glitch, and the number of clean processes since the previous glitch) so the
glitch's pattern can be read back later. The merged report is written to
`<out-dir>/merged.json`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401
from report_evaluation import merge_evaluations

GLITCH_MARKER = "not drawing the robot"
EVAL_SCRIPT = Path(__file__).resolve().parent / "eval_policy.py"


def shard_ranges(seed: int, episodes: int, shard_size: int) -> list[tuple[int, int]]:
    """`(first_seed, count)` per shard covering `episodes` seeds from `seed`."""
    if episodes < 1 or shard_size < 1:
        raise ValueError("episodes and shard_size must be positive")
    return [
        (start, min(shard_size, seed + episodes - start))
        for start in range(seed, seed + episodes, shard_size)
    ]


def run_shard(
    start: int, count: int, out: Path, passthrough: list[str]
) -> tuple[int, bool, float]:
    """Run one shard; return (exit code, hit the render glitch, seconds)."""
    command = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--seed",
        str(start),
        "--episodes",
        str(count),
        "--json-out",
        str(out),
        *passthrough,
    ]
    began = time.monotonic()
    completed = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    out.with_suffix(".log").write_text(completed.stdout, encoding="utf-8")
    # Kit swallows the exit status of a Python error raised under the app (a
    # glitch at start-up exits 0), so the marker alone says it happened.
    glitch = GLITCH_MARKER in completed.stdout
    return completed.returncode, glitch, time.monotonic() - began


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--shard-size", type=int, default=10)
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument(
        "--cooldown",
        type=float,
        default=60.0,
        help="seconds to wait before every process after the first; Isaac starts "
        "within a minute of the previous process glitched in 8 of 17 cases and "
        "starts after a 90 s pause in 1 of 5 (Fisher p = 0.36, so only suggestive)",
    )
    ap.add_argument("passthrough", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    passthrough = [a for a in args.passthrough if a != "--"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "attempts.jsonl"
    clean_since_glitch = 0
    parts = []
    for start, count in shard_ranges(args.seed, args.episodes, args.shard_size):
        out = args.out_dir / f"shard-{start}.json"
        for attempt in range(1, args.retries + 2):
            if out.exists():
                out.unlink()
            if args.cooldown > 0 and (parts or attempt > 1):
                time.sleep(args.cooldown)
            stamp = datetime.now().isoformat(timespec="seconds")
            code, glitch, seconds = run_shard(start, count, out, passthrough)
            record = {
                "shard": start,
                "attempt": attempt,
                "started": stamp,
                "seconds": round(seconds, 1),
                "exit_code": code,
                "render_glitch": glitch,
                "clean_since_last_glitch": clean_since_glitch,
            }
            with log_path.open("a", encoding="utf-8") as log:
                log.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            if out.exists():
                clean_since_glitch += 1
                parts.append(json.loads(out.read_text(encoding="utf-8")))
                break
            if glitch:
                clean_since_glitch = 0
            elif code != 0 or not out.exists():
                raise SystemExit(
                    f"shard {start} failed for another reason; see {out.with_suffix('.log')}"
                )
        else:
            raise SystemExit(
                f"shard {start} still failing after {args.retries} retries"
            )

    merged = parts[0] if len(parts) == 1 else merge_evaluations(parts)
    (args.out_dir / "merged.json").write_text(
        json.dumps(merged, indent=2), encoding="utf-8"
    )
    summary = merged["summary"]
    print(
        f"merged {summary['episodes']} episodes: "
        f"{summary['success_count']}/{summary['episodes']} succeeded -> {args.out_dir / 'merged.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
