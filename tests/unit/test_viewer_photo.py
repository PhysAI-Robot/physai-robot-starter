import sys
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
