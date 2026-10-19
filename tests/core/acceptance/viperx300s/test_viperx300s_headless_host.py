"""`run_sim.py --serve` on the ViperX 300 S: the web viewer comes up, steps, and serves both cameras."""

import json
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip(
    "fastapi", reason="fastapi is a base dependency; reinstall with uv sync"
)

REPO_ROOT = Path(__file__).resolve().parents[4]

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.assets,
    pytest.mark.slow,
    pytest.mark.skipif(
        not (REPO_ROOT / "assets" / "viperx300s" / "vx300s.xml").exists(),
        reason="run `python scripts/fetch_assets.py --robot viperx300s`",
    ),
]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def get(url: str, timeout: float = 5.0) -> tuple[int, str, bytes]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return (
            response.status,
            response.headers.get("content-type", ""),
            response.read(),
        )


def wait_until_serving(base: str, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(f"host exited early with code {process.returncode}")
        try:
            get(f"{base}/api/robots", timeout=2.0)
            return
        except OSError:
            time.sleep(0.5)
    raise AssertionError(f"host did not start serving within {timeout:.0f}s")


@pytest.mark.parametrize(
    ("manifest", "policy"),
    [
        ("configs/manifests/viperx300s_single_cube_fixed_place.yaml", "constant"),
        ("configs/manifests/viperx300s_single_cube_randomized_place.yaml", "constant"),
    ],
)
def test_headless_host_serves_the_viperx300s_with_both_cameras(manifest, policy):
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_sim.py"),
        "--manifest",
        manifest,
        "--policy",
        policy,
        "--serve",
        "--port",
        str(port),
        "--seed",
        "0",
    ]
    with tempfile.TemporaryFile("w+") as log:
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True
        )
        try:
            wait_until_serving(base, process, timeout=120.0)
            robots = json.loads(get(f"{base}/api/robots")[2])
            assert robots, "the host should report its registered robot"

            first = json.loads(get(f"{base}/api/state")[2])["step"]
            time.sleep(1.5)
            second = json.loads(get(f"{base}/api/state")[2])["step"]
            assert second > first, "the physics loop should be advancing"

            for camera in ("front", "wrist"):
                status, content_type, body = get(f"{base}/api/camera/{camera}.jpg")
                assert status == 200 and content_type == "image/jpeg"
                assert body[:2] == b"\xff\xd8", f"{camera} is not a JPEG"
        finally:
            if sys.platform == "win32":
                process.terminate()
            else:
                process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                pytest.fail("host did not stop within 30s of SIGINT")
