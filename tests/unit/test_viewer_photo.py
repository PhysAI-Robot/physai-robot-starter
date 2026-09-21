import sys
import threading
import types
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


@pytest.fixture
def tk_module():
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"no display for Tk: {exc}")
    root.withdraw()
    yield tk
    root.destroy()


def test_photo_carries_the_frame_pixels_without_a_temporary_file(
    monkeypatch, tk_module
):
    """The viewer used to write each frame to a temp file and give Tk its name.

    On Windows a temporary file that is still open cannot be opened again by
    name, so the first render raised "permission denied" and the viewer died.
    """
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import run_sim

    frame = np.zeros((4, 6, 3), np.uint8)
    frame[..., 0] = 200
    frame[1, 2] = (10, 20, 30)
    frame[3, 5] = (250, 251, 252)

    photo = run_sim.SingleWindowViewer._photo(
        types.SimpleNamespace(_tk=tk_module), frame
    )

    assert (photo.width(), photo.height()) == (6, 4)
    assert photo.get(0, 0) == (200, 0, 0)
    assert photo.get(2, 1) == (10, 20, 30)
    assert photo.get(5, 3) == (250, 251, 252)
    assert "tempfile" not in vars(sys.modules["run_sim"])


def test_render_tolerates_a_camera_frame_that_has_not_arrived_yet(monkeypatch):
    """Camera capture runs on a worker thread started concurrently with the
    physics thread; the first render() used to crash with
    "ValueError: unknown camera 'front'" if that worker had not produced its
    first frame yet.
    """
    tk = pytest.importorskip("tkinter")
    mujoco = pytest.importorskip("mujoco")
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import run_sim

    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><geom type='box' size='0.1 0.1 0.1'/></worldbody></mujoco>"
    )

    class FakeHost:
        def __init__(self, model) -> None:
            self.model = model
            self.data = mujoco.MjData(model)
            self.physics_lock = threading.Lock()
            self.paused = False
            self.robot_name = "test"
            self.images: dict[str, np.ndarray] = {}

        def camera_image(self, name):
            try:
                return self.images[name]
            except KeyError as exc:
                raise ValueError(f"unknown camera {name!r}") from exc

        def latest_state(self):
            return None

    host = FakeHost(model)
    try:
        app = run_sim.SingleWindowViewer(host, ["front"])
    except tk.TclError as exc:
        pytest.skip(f"no display for Tk: {exc}")
    try:
        app.render()  # must not raise even though "front" has no frame yet

        host.images["front"] = np.zeros((4, 4, 3), dtype=np.uint8)
        app.render()

        assert app.camera_photos["front"] is not None
    finally:
        app.close()
