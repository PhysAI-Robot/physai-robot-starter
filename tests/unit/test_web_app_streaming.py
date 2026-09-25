import asyncio

import anyio
import numpy as np
import pytest

from physai.robots import RobotSpec
from physai.web.host import Host
from tests.support.fakes import FakeRobotPort

pytest.importorskip("fastapi")
pytest.importorskip("httpx2")

from fastapi.testclient import TestClient  # noqa: E402

from physai.web.app import create_app  # noqa: E402


def make_host_with_camera(name: str = "front") -> Host:
    spec = RobotSpec(name="test", kind="test", joint_names=("joint",))
    host = Host.for_robot(FakeRobotPort(spec), robot_name="test")
    host._cameras.put(f"{host.robot_name}:{name}", np.zeros((4, 4, 3), dtype=np.uint8))
    return host


async def _read_first_stream_chunk(app, path: str, timeout: float = 5.0):
    """Read one chunk from an infinite ASGI stream, then cancel it.

    The camera stream endpoint yields forever by design (a real MJPEG feed
    only stops when the client disconnects). Both Starlette's TestClient and
    httpx2's ASGITransport await the whole ASGI response before returning
    anything, so they hang forever against this endpoint; driving the ASGI
    protocol directly and cancelling the app task once one chunk arrives is
    the only way to exercise it without hanging.
    """
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [],
        "server": ("testserver", 80),
        "client": ("testclient", 123),
        "root_path": "",
    }
    status: dict = {}
    got_chunk = asyncio.Event()
    chunks: list[bytes] = []
    request_sent = False
    never = asyncio.Event()

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": b"", "more_body": False}
        # No real disconnect yet; block like a real receive channel would
        # instead of busy-looping Starlette's listen_for_disconnect().
        await never.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.start":
            status["code"] = message["status"]
            status["headers"] = dict(
                (k.decode(), v.decode()) for k, v in message.get("headers", [])
            )
        elif message["type"] == "http.response.body":
            body = message.get("body", b"")
            if body:
                chunks.append(body)
                got_chunk.set()

    task = asyncio.ensure_future(app(scope, receive, send))
    try:
        await asyncio.wait_for(got_chunk.wait(), timeout=timeout)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    return status, chunks[0] if chunks else b""


def test_camera_streams_are_mjpeg_and_unknown_cameras_are_404():
    # a policy-published debug overlay is named "<camera>:detections"; the colon
    # must survive FastAPI's path routing end to end
    for camera in ("front", "front:detections"):
        app = create_app(host=make_host_with_camera(camera))

        status, chunk = anyio.run(
            _read_first_stream_chunk, app, f"/api/camera/{camera}/stream"
        )

        assert status["code"] == 200, camera
        assert "multipart/x-mixed-replace" in status["headers"]["content-type"]
        assert b"\xff\xd8" in chunk  # a JPEG frame

    client = TestClient(create_app(host=make_host_with_camera("front")))
    assert client.get("/api/camera/missing/stream").status_code == 404
