import numpy as np
import pytest

from physai.robots import RobotSpec
from physai.web.runtime import SimulationHost
from tests.support.fakes import FakeRobotPort

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from physai.web.app import create_app  # noqa: E402


def make_host_with_camera(name: str = "front") -> SimulationHost:
    spec = RobotSpec(name="test", kind="test", joint_names=("joint",))
    host = SimulationHost(FakeRobotPort(spec), robot_name="test")
    host._camera_images[name] = np.zeros((4, 4, 3), dtype=np.uint8)
    return host


def test_camera_stream_returns_multipart_mjpeg():
    app = create_app(host=make_host_with_camera("front"))
    client = TestClient(app)

    with client.stream("GET", "/api/camera/front/stream") as response:
        assert response.status_code == 200
        assert "multipart/x-mixed-replace" in response.headers["content-type"]
        chunk = next(response.iter_bytes())
        assert b"\xff\xd8" in chunk


def test_camera_stream_unknown_camera_returns_404():
    app = create_app(host=make_host_with_camera("front"))
    client = TestClient(app)

    response = client.get("/api/camera/missing/stream")

    assert response.status_code == 404
