"""Download the SO-101 robot description (MJCF + meshes) into assets/so101/.

Source: TheRobotStudio/SO-ARM100, Simulation/SO101/
The directory is listed via the GitHub contents API so we never hardcode mesh
filenames (they change between calibration revisions).

    python scripts/fetch_assets.py
    python scripts/fetch_assets.py --force
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = "TheRobotStudio/SO-ARM100"
SUBDIR = "Simulation/SO101"
REF = "main"
DEST = Path(__file__).resolve().parents[1] / "assets" / "so101"

API = "https://api.github.com/repos/{repo}/contents/{path}?ref={ref}"


def _get_json(url: str) -> list[dict]:
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                               "User-Agent": "physai-robot-starter"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def _download(url: str, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "physai-robot-starter"})
    with urllib.request.urlopen(req, timeout=120) as r:
        blob = r.read()
    dest.write_bytes(blob)
    return len(blob)


def walk(path: str, out_root: Path, rel: Path, force: bool) -> tuple[int, int]:
    """Recursively mirror a GitHub directory. Returns (files, bytes)."""
    entries = _get_json(API.format(repo=REPO, path=path, ref=REF))
    n_files = n_bytes = 0
    for e in entries:
        name = e["name"]
        if e["type"] == "dir":
            f, b = walk(f"{path}/{name}", out_root, rel / name, force)
            n_files, n_bytes = n_files + f, n_bytes + b
            continue
        if e["type"] != "file":
            continue
        dest = out_root / rel / name
        if dest.exists() and not force:
            print(f"  skip   {rel / name}")
            continue
        size = _download(e["download_url"], dest)
        n_files += 1
        n_bytes += size
        print(f"  get    {rel / name}  ({size / 1024:.0f} KB)")
    return n_files, n_bytes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    ap.add_argument("--dest", type=Path, default=DEST)
    args = ap.parse_args()

    print(f"Fetching {REPO}/{SUBDIR} @ {REF}")
    print(f"  -> {args.dest}")
    try:
        n, b = walk(SUBDIR, args.dest, Path("."), args.force)
    except urllib.error.HTTPError as exc:
        print(f"\nGitHub API error {exc.code}: {exc.reason}", file=sys.stderr)
        if exc.code == 403:
            print("Rate limited (60 req/h unauthenticated). Wait an hour, or clone the "
                  "repo manually and copy Simulation/SO101 into assets/so101/.",
                  file=sys.stderr)
        return 1

    print(f"\nDone: {n} new files, {b / 1e6:.1f} MB")

    xmls = sorted(p.name for p in args.dest.rglob("*.xml"))
    if not xmls:
        print("WARNING: no .xml found — the upstream layout may have changed.", file=sys.stderr)
        return 1
    print("MJCF files available:")
    for x in xmls:
        print(f"  {x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
