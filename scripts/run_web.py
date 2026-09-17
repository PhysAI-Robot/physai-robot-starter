"""Run the headless MuJoCo browser viewer.

Usage: uv run --extra web python scripts/run_web.py --robot so101
"""

from __future__ import annotations

import argparse
import os

import _bootstrap  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot",
        dest="robots",
        action="append",
        help="registered robot to start; repeat for a multi-robot fleet",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="run the scripted SO-101 pick-and-place policy in the browser viewer",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # The browser owns presentation; MuJoCo only renders offscreen camera frames.
    os.environ.setdefault("MUJOCO_GL", "egl")

    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("install web dependencies with: uv sync --extra web") from exc

    from physai.web.app import create_app

    uvicorn.run(
        create_app(
            robot_name=(args.robots or ["so101"])[0],
            robot_names=tuple(args.robots or ["so101"]),
            demo=args.demo,
        ),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
