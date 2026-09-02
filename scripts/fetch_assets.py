"""Download the SO-101 robot description (MJCF + meshes) into assets/so101/.

Sources:
    TheRobotStudio/SO-ARM100, Simulation/SO101/
    turtlebot/turtlebot4, turtlebot4_description/
The directory is listed via the GitHub contents API so we never hardcode mesh
filenames (they change between calibration revisions).

    python scripts/fetch_assets.py --robot so101
    python scripts/fetch_assets.py --robot turtlebot4
    python scripts/fetch_assets.py --robot so101 --force
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AssetSource:
    """Remote robot description source consumed by the generic downloader."""

    repository: str
    path: str
    ref: str
    description_globs: tuple[str, ...] = ("*.xml", "*.xacro")
    include: tuple[str, ...] = ()


SOURCES: dict[str, AssetSource] = {
    "so101": AssetSource("TheRobotStudio/SO-ARM100", "Simulation/SO101", "main"),
    "turtlebot4": AssetSource(
        "narcispr/turtlebot4_mujoco", "", "main",
        description_globs=("*.xml",),
        include=("turtlebot4.xml", "assets/meshes/*.stl", "assets/meshes/*.obj"),
    ),
}
DEST_ROOT = Path(__file__).resolve().parents[1] / "assets"

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


def walk(path: str, out_root: Path, rel: Path, force: bool,
         repo: str, ref: str, include: tuple[str, ...] = ()) -> tuple[int, int]:
    """Recursively mirror a GitHub directory. Returns (files, bytes)."""
    entries = _get_json(API.format(repo=repo, path=path, ref=ref))
    n_files = n_bytes = 0
    for e in entries:
        name = e["name"]
        if e["type"] == "dir":
            f, b = walk(f"{path}/{name}", out_root, rel / name, force,
                        repo, ref, include)
            n_files, n_bytes = n_files + f, n_bytes + b
            continue
        if e["type"] != "file":
            continue
        relative_path = (rel / name).as_posix()
        if include and not any(Path(relative_path).match(pattern) for pattern in include):
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
    ap.add_argument("--robot", choices=sorted(SOURCES), default="so101")
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    ap.add_argument("--dest", type=Path)
    args = ap.parse_args()

    source = SOURCES[args.robot]
    dest = args.dest or DEST_ROOT / args.robot
    print(f"Fetching {source.repository}/{source.path} @ {source.ref}")
    print(f"  -> {dest}")
    try:
        n, b = walk(source.path, dest, Path("."), args.force,
                 source.repository, source.ref, source.include)
    except urllib.error.HTTPError as exc:
        print(f"\nGitHub API error {exc.code}: {exc.reason}", file=sys.stderr)
        if exc.code == 403:
            print("Rate limited (60 req/h unauthenticated). Wait an hour, or "
                  "download the source repository manually.", file=sys.stderr)
        return 1

    print(f"\nDone: {n} new files, {b / 1e6:.1f} MB")

    descriptions = sorted(
        p.name for pattern in source.description_globs for p in dest.rglob(pattern)
    )
    if not descriptions:
        print("WARNING: no XML/Xacro description found — the upstream layout may have changed.", file=sys.stderr)
        return 1
    print("Robot description files available:")
    for description in descriptions:
        print(f"  {description}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
