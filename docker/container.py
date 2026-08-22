"""Build, start, and stop the project container.

    python docker/container.py build
    python docker/container.py start
    python docker/container.py shell
    python docker/container.py stop

The CUDA base image is used when nvidia-smi is present, otherwise the plain
Ubuntu one. Force either with --gpu / --cpu; a MacBook needs --cpu.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

DOCKER_DIR = Path(__file__).resolve().parent
COMPOSE = DOCKER_DIR / "docker-compose.yml"
COMPOSE_GPU = DOCKER_DIR / "docker-compose.gpu.yml"

CUDA_BASE = "nvidia/cuda:12.8.1-devel-ubuntu22.04"
CPU_BASE = "ubuntu:22.04"


def gpu_available() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    info = subprocess.run(["docker", "info", "--format", "{{json .Runtimes}}"],
                          capture_output=True, text=True)
    return "nvidia" in info.stdout


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
    mode.add_argument("--gpu", action="store_true", help="force the CUDA image")
    mode.add_argument("--cpu", action="store_true", help="force the CPU-only image")
    ap.add_argument("--no-cache", action="store_true", help="build without the layer cache")
    args, extra = ap.parse_known_args()

    gpu = args.gpu or (not args.cpu and gpu_available())
    if gpu and not gpu_available():
        print("warning: nvidia-smi not found; --gpu needs the NVIDIA Container Toolkit",
              file=sys.stderr)

    print(f"mode: {'gpu' if gpu else 'cpu'} (base image {CUDA_BASE if gpu else CPU_BASE})")

    if args.command == "build":
        return compose(["build", *(["--no-cache"] if args.no_cache else []), *extra], gpu)
    if args.command == "start":
        return compose(["up", "-d", *extra], gpu)
    if args.command == "stop":
        return compose(["down", *extra], gpu)
    return compose(["exec", "physai", "bash", *extra], gpu)


if __name__ == "__main__":
    raise SystemExit(main())
