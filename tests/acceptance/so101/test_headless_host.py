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

pytest.importorskip("fastapi", reason="install the web extra: uv sync --extra web")

REPO_ROOT = Path(__file__).resolve().parents[3]
CAMERA_VARIANT = REPO_ROOT / "assets" / "so101" / "so101_new_calib_camera.xml"

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.assets,
    pytest.mark.slow,
    pytest.mark.skipif(
        not CAMERA_VARIANT.exists(),
        reason="run `python scripts/fetch_assets.py` to download the camera variant",
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


def test_headless_host_serves_the_web_viewer_without_a_desktop_window():
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_sim.py"),
        "--config",
        "configs/tasks/so101/pick_place.yaml",
        "--policy",
        "visual_servo",
        "--headless",
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
            wait_until_serving(base, process, timeout=90.0)

            status, content_type, body = get(f"{base}/")
            assert status == 200 and "text/html" in content_type

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

            with urllib.request.urlopen(
                f"{base}/api/camera/front/stream", timeout=5.0
            ) as response:
                assert response.status == 200
                assert "multipart/x-mixed-replace" in response.headers.get(
                    "content-type", ""
                )
                chunk = response.read(2048)
                assert b"\xff\xd8" in chunk, "stream did not contain a JPEG frame"
        finally:
            if sys.platform == "win32":
                # Windows cannot deliver SIGINT to a child that is not attached
                # to this console; tests/unit/test_simulation_host.py covers the
                # ordered shutdown, so only make sure the process goes away.
                process.terminate()
            else:
                process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                pytest.fail("host did not stop within 30s of SIGINT")
            log.seek(0)
            output = log.read()

    if sys.platform != "win32":
        assert process.returncode == 0, f"unclean shutdown:\n{output}"
