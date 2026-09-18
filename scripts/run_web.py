"""Open the browser viewer for an existing simulation host.

Usage: uv run --extra web python scripts/run_web.py --connect http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import webbrowser
from urllib.parse import urlparse

import _bootstrap  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--connect",
        default="http://127.0.0.1:8000",
        help="URL of the running simulation host",
    )
    args = parser.parse_args()

    parsed = urlparse(args.connect)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        parser.error("--connect must be an http(s) URL")
    print(f"Opening shared simulation host: {args.connect}")
    if not webbrowser.open(args.connect):
        print("Open the URL above in a browser.")


if __name__ == "__main__":
    main()
