"""Build, start, and stop the project container.

    python docker/container.py build
    python docker/container.py start
    python docker/container.py shell
    python docker/container.py stop

The CUDA base image is used by default; pass --cpu to build against the plain
Ubuntu one, which is what a MacBook needs. The mode is chosen at build time and
recorded, so start/stop/shell reuse it.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

DOCKER_DIR = Path(__file__).resolve().parent
COMPOSE = DOCKER_DIR / "docker-compose.yml"
COMPOSE_GPU = DOCKER_DIR / "docker-compose.gpu.yml"
MODE_FILE = DOCKER_DIR / ".mode"

CUDA_BASE = "nvidia/cuda:12.8.1-devel-ubuntu24.04"
CPU_BASE = "ubuntu:24.04"


def compose(args: list[str], gpu: bool) -> int:
    files = ["-f", str(COMPOSE)]
    if gpu:
        files += ["-f", str(COMPOSE_GPU)]
    cmd = ["docker", "compose", *files, *args]
    env = dict(os.environ, BASE_IMAGE=CUDA_BASE if gpu else CPU_BASE)
    print("+", " ".join(cmd))
    return subprocess.run(cmd, env=env).returncode


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("command", choices=["build", "start", "stop", "shell"])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--gpu", action="store_true", help="build the CUDA image (default)")
    mode.add_argument("--cpu", action="store_true", help="build the CPU-only image")
    ap.add_argument("--no-cache", action="store_true", help="build without the layer cache")
    args, extra = ap.parse_known_args()

    if args.command == "build":
        gpu = not args.cpu
    else:
        if args.gpu or args.cpu:
            ap.error("--gpu/--cpu apply to build only; rebuild to switch mode")
        gpu = MODE_FILE.read_text().strip() != "cpu" if MODE_FILE.exists() else True

    print(f"mode: {'gpu' if gpu else 'cpu'} (base image {CUDA_BASE if gpu else CPU_BASE})")

    if args.command == "build":
        rc = compose(["build", *(["--no-cache"] if args.no_cache else []), *extra], gpu)
        if rc == 0:
            MODE_FILE.write_text("gpu" if gpu else "cpu")
        return rc
    if args.command == "start":
        return compose(["up", "-d", *extra], gpu)
    if args.command == "stop":
        return compose(["down", *extra], gpu)
    return compose(["exec", "physai", "bash", *extra], gpu)


if __name__ == "__main__":
    raise SystemExit(main())
