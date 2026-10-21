"""Download a robot description (MJCF + meshes) into assets/<robot>/, and its USD for Isaac Sim.

Sources:
    TheRobotStudio/SO-ARM100, Simulation/SO101/
    turtlebot/turtlebot4, turtlebot4_description/
    google-deepmind/mujoco_menagerie, trossen_vx300s/   (robot "viperx300s")
The directory is listed via the GitHub contents API so we never hardcode mesh
filenames (they change between calibration revisions).

    python scripts/fetch_assets.py --robot so101
    python scripts/fetch_assets.py --robot turtlebot4
    python scripts/fetch_assets.py --robot viperx300s
    python scripts/fetch_assets.py --robot so101 --force

When Isaac Sim is installed (`uv sync --extra isaac`; set OMNI_KIT_ACCEPT_EULA=YES) the robots that have an
Isaac Sim env (so101, viperx300s) also get their USD, written to assets/<robot>/usd/ by Isaac's own importer.
Isaac Sim runs read that USD and stop with an error that points back here when it is missing or was written
by another Isaac Sim version, so fetch again (--force rewrites it) after installing or upgrading isaacsim.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import _bootstrap  # noqa: F401
from _cli import add_robot, new_parser


@dataclass(frozen=True)
class AssetSource:
    """Remote robot description source consumed by the generic downloader."""

    repository: str
    path: str
    ref: str
    description_globs: tuple[str, ...] = ("*.xml", "*.xacro")
    include: tuple[str, ...] = ()


SOURCES: dict[str, AssetSource] = {
    # Pinned to a commit, not a branch: an upstream push must not change what
    # this project fetches. Bump deliberately (see THIRD_PARTY_NOTICES.md) and
    # re-run the full test suite plus a `visual_servo` eval before adopting a
    # new commit.
    "so101": AssetSource(
        "TheRobotStudio/SO-ARM100",
        "Simulation/SO101",
        "5f6d2b876a53a4872e405b991dd925556c9e38a4",
    ),
    "turtlebot4": AssetSource(
        "narcispr/turtlebot4_mujoco",
        "",
        "e5d772caf426179b3d93eb91741965529239ae80",
        description_globs=("*.xml",),
        include=("turtlebot4.xml", "assets/meshes/*.stl", "assets/meshes/*.obj"),
    ),
    # The ViperX 300 S arm of the ALOHA follower (single arm, stock gripper), from
    # MuJoCo Menagerie. Pinned like the others; see THIRD_PARTY_NOTICES.md.
    "viperx300s": AssetSource(
        "google-deepmind/mujoco_menagerie",
        "trossen_vx300s",
        "62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35",
        include=("vx300s.xml", "assets/*.stl", "assets/*.png", "LICENSE"),
    ),
}
ISAAC_ROBOTS = ("so101", "viperx300s")  # the robots with an Isaac Sim env
DEST_ROOT = Path(__file__).resolve().parents[1] / "assets"

API = "https://api.github.com/repos/{repo}/contents/{path}?ref={ref}"


def _api_headers() -> dict[str, str]:
    """Headers for api.github.com, authenticated when a token is available.

    Unauthenticated requests share a 60 per hour limit per address, which
    concurrent CI jobs and shared cloud workspaces exhaust. A token raises it
    to a per-token limit. It is sent only to the API listing requests, never to
    the file downloads.
    """
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "physai-robot-starter",
    }
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _get_json(url: str) -> list[dict]:
    req = urllib.request.Request(url, headers=_api_headers())
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def _download(url: str, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "physai-robot-starter"})
    with urllib.request.urlopen(req, timeout=120) as r:
        blob = r.read()
    dest.write_bytes(blob)
    return len(blob)


def walk(
    path: str,
    out_root: Path,
    rel: Path,
    force: bool,
    repo: str,
    ref: str,
    include: tuple[str, ...] = (),
) -> tuple[int, int]:
    """Recursively mirror a GitHub directory. Returns (files, bytes)."""
    entries = _get_json(API.format(repo=repo, path=path, ref=ref))
    n_files = n_bytes = 0
    for e in entries:
        name = e["name"]
        if e["type"] == "dir":
            f, b = walk(
                f"{path}/{name}", out_root, rel / name, force, repo, ref, include
            )
            n_files, n_bytes = n_files + f, n_bytes + b
            continue
        if e["type"] != "file":
            continue
        relative_path = (rel / name).as_posix()
        if include and not any(
            Path(relative_path).match(pattern) for pattern in include
        ):
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


def dump_usd(robot: str, dest: Path, force: bool) -> None:
    """Write the robot's USD for Isaac Sim into `dest/usd/`, when Isaac Sim is installed."""
    if robot not in ISAAC_ROBOTS:
        return
    if importlib.util.find_spec("isaacsim") is None:
        print(
            "\nIsaac Sim is not installed, so no USD was written. Isaac Sim runs need it: "
            "`uv sync --extra isaac`, then run this again."
        )
        return
    from physai.robots.description import load_robot_description
    from physai.sim.isaac import current_usd, dump_robot_usd, ensure_simulation_app

    module = importlib.import_module(f"physai.robots.{robot}")
    description = load_robot_description(
        Path(module.__file__).parent / "description.yaml"
    )
    existing = current_usd(description, dest)
    if existing is not None and not force:
        print(f"\nIsaac Sim USD is current: {existing}")
        return
    print("\nImporting the robot into USD with Isaac Sim (this starts the simulator)")
    ensure_simulation_app()
    print(f"Isaac Sim USD: {dump_robot_usd(description, dest)}")


def main() -> int:
    ap = new_parser(__doc__)
    add_robot(ap, choices=sorted(SOURCES), default="so101")
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    ap.add_argument(
        "--out",
        type=Path,
        help="folder to download into (default: assets/<robot>)",
    )
    args = ap.parse_args()

    source = SOURCES[args.robot]
    dest = args.out or DEST_ROOT / args.robot
    print(f"Fetching {source.repository}/{source.path} @ {source.ref}")
    print(f"  -> {dest}")
    try:
        n, b = walk(
            source.path,
            dest,
            Path("."),
            args.force,
            source.repository,
            source.ref,
            source.include,
        )
    except urllib.error.HTTPError as exc:
        print(f"\nGitHub API error {exc.code}: {exc.reason}", file=sys.stderr)
        if exc.code == 403:
            print(
                "Rate limited (60 req/h unauthenticated). Set GITHUB_TOKEN to "
                "authenticate, wait an hour, or download the source repository "
                "manually.",
                file=sys.stderr,
            )
        return 1

    print(f"\nDone: {n} new files, {b / 1e6:.1f} MB")

    descriptions = sorted(
        p.name for pattern in source.description_globs for p in dest.rglob(pattern)
    )
    if not descriptions:
        print(
            "WARNING: no XML/Xacro description found — the upstream layout may have changed.",
            file=sys.stderr,
        )
        return 1
    print("Robot description files available:")
    for description in descriptions:
        print(f"  {description}")
    dump_usd(args.robot, dest, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
