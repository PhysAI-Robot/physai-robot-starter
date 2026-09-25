"""Cached camera frames, rendered on a worker thread decoupled from physics.

Rendering costs far more than a physics tick, so the worker renders named
cameras on its own cadence from a private copy of the simulation state and
publishes the latest frame of each into a cache. Everything else (HTTP
requests, policies, recording) only reads that cache; nothing on the physics
thread or a request handler ever renders. The worker owns its renderer and
its OpenGL context: they are created and closed on its thread.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence

import mujoco
import numpy as np


class CameraFeed:
    """A frame cache keyed by ``"<instance>:<camera>"`` plus its render worker."""

    # Capture cadence. Must match `app.py`'s _CAMERA_STREAM_PERIOD.
    PERIOD = 1.0 / 30

    def __init__(self) -> None:
        self._images: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._request = threading.Event()
        self._ready = threading.Event()

    # -- cache ------------------------------------------------------------
    def put(self, key: str, image: np.ndarray) -> None:
        with self._lock:
            self._images[key] = image

    def get(self, key: str) -> np.ndarray | None:
        """A copy of the latest frame under ``key``, or None."""
        with self._lock:
            image = self._images.get(key)
            return None if image is None else image.copy()

    def with_prefix(self, prefix: str) -> dict[str, np.ndarray]:
        """The frames whose key starts with ``prefix``, keyed by the rest."""
        with self._lock:
            return {
                key[len(prefix) :]: image
                for key, image in self._images.items()
                if key.startswith(prefix)
            }

    # -- worker -----------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._thread is not None

    def start(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        physics_lock: threading.Lock,
        stop: threading.Event,
        cameras: Sequence[tuple[str, str]],
        size: tuple[int, int],
    ) -> None:
        """Start rendering ``(cache key, MuJoCo camera name)`` pairs; once only."""
        if self._thread is not None:
            return
        scratch = mujoco.MjData(model)
        self._thread = threading.Thread(
            target=self._run,
            args=(model, data, scratch, physics_lock, stop, tuple(cameras), size),
            name="physai-camera",
            daemon=True,
        )
        self._thread.start()

    def request(self, *, wait: bool = False) -> None:
        """Ask for a capture now instead of at the next tick, optionally waiting."""
        if self._thread is None:
            return
        self._ready.clear()
        self._request.set()
        if wait:
            self._ready.wait(timeout=2.0)

    def join(self, timeout: float = 2.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _run(self, model, data, scratch, physics_lock, stop, cameras, size) -> None:
        width, height = size
        renderer = mujoco.Renderer(model, height=height, width=width)
        next_capture = 0.0
        try:
            while not stop.is_set():
                timeout = max(0.0, next_capture - time.monotonic())
                requested = self._request.wait(timeout=timeout)
                self._request.clear()
                if stop.is_set():
                    break
                if not requested and time.monotonic() < next_capture:
                    continue
                with physics_lock:
                    mujoco.mj_copyData(scratch, model, data)
                for key, mujoco_name in cameras:
                    renderer.update_scene(scratch, camera=mujoco_name)
                    self.put(key, np.asarray(renderer.render(), dtype=np.uint8).copy())
                next_capture = time.monotonic() + self.PERIOD
                self._ready.set()
        finally:
            renderer.close()


__all__ = ["CameraFeed"]
