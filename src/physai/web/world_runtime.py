"""Web runtime for multiple robot instances in one MuJoCo world."""

from __future__ import annotations

import threading
import time
from io import BytesIO
from typing import Any

import imageio.v3 as iio
import mujoco
import numpy as np

from ..contracts import Action
from ..robots.registry import create_shared_instance
from ..sim.world import RobotInstanceConfig, SharedWorld
from .telemetry import build_scene_manifest, build_state_snapshot

# Every per-robot instance (SO101SharedInstance, TurtleBot4SharedInstance, ...)
# is built through robots.registry.create_shared_instance() below, keyed by
# each config's robot_name — this class never branches on a robot name itself.


class SharedWorldHost:
    """One web host for a composite world and its per-instance controllers."""

    _CAMERA_PERIOD = 0.2

    def __init__(self, world: SharedWorld, instances: tuple[RobotInstanceConfig, ...]):
        self.world = world
        self.instances = {
            config.instance_id: create_shared_instance(config.robot_name, world, config)
            for config in instances
        }
        self._lock = threading.Lock()
        self._physics_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._paused = False
        self._state: dict[str, Any] | None = None
        self._control_owner: dict[str, str] = {}
        self._control_deadline: dict[str, float] = {}
        self._pending: dict[str, Action] = {}
        self._camera_images: dict[str, np.ndarray] = {}
        self._camera_data = mujoco.MjData(self.model)
        self._camera_thread: threading.Thread | None = None
        self._camera_request = threading.Event()
        self._camera_ready = threading.Event()
        self._renderer: mujoco.Renderer | None = None

    @property
    def model(self):
        return self.world.model

    @property
    def data(self):
        return self.world.data

    @property
    def physics_lock(self):
        return self._physics_lock

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def control_hz(self) -> float:
        return float(self.world.control_hz)

    def scene(self) -> dict[str, Any]:
        return build_scene_manifest(
            self.model,
            robot="shared",
            instance_prefixes={
                instance_id: instance.prefix
                for instance_id, instance in self.instances.items()
            },
        )

    def start(self) -> None:
        if self._thread is not None:
            return
        self._start_camera_thread()
        self._thread = threading.Thread(
            target=self._run, name="physai-shared-world", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is None:
            self.world.close()
            return
        self._thread.join(timeout=2.0)
        if self._camera_thread is not None:
            self._camera_thread.join(timeout=2.0)
        self.world.close()

    def camera_jpeg(self, instance_id: str, name: str) -> bytes:
        camera_key = f"{instance_id}:{name}"
        with self._lock:
            try:
                image = self._camera_images[camera_key].copy()
            except KeyError as exc:
                raise ValueError(
                    f"unknown camera {name!r} for robot instance {instance_id!r}"
                ) from exc
        buffer = BytesIO()
        iio.imwrite(buffer, image, extension=".jpg", quality=82)
        return buffer.getvalue()

    def reset(self) -> None:
        self._start_camera_thread()
        with self._physics_lock:
            self.world.reset()
            for instance in self.instances.values():
                instance.reset()
            import mujoco

            mujoco.mj_forward(self.model, self.data)
            self._pending.clear()
            self._publish()
        self._request_camera_capture(wait=True)

    def set_paused(self, paused: bool) -> None:
        self._paused = paused

    def release_control(self, source: str) -> None:
        with self._lock:
            for instance_id, owner in list(self._control_owner.items()):
                if owner == source:
                    self._control_owner.pop(instance_id, None)
                    self._control_deadline.pop(instance_id, None)
                    self._pending.pop(instance_id, None)

    def submit(
        self, instance_id: str, action: Action, *, source: str = "local"
    ) -> None:
        if instance_id not in self.instances:
            raise ValueError(f"unknown robot instance {instance_id!r}")
        with self._physics_lock:
            action = self.instances[instance_id].prepare_action(action)
        now = time.monotonic()
        with self._lock:
            owner = self._control_owner.get(instance_id)
            if (
                owner not in (None, source)
                and now < self._control_deadline[instance_id]
            ):
                raise PermissionError(
                    f"robot instance {instance_id!r} is controlled by another client"
                )
            self._control_owner[instance_id] = source
            self._control_deadline[instance_id] = now + 0.35
            self._pending[instance_id] = action

    def latest_state(self) -> dict[str, Any] | None:
        with self._lock:
            return None if self._state is None else dict(self._state)

    def _run(self) -> None:
        self.reset()
        period = 1.0 / self.control_hz
        while not self._stop.is_set():
            started = time.monotonic()
            if not self._paused:
                with self._physics_lock:
                    for instance_id, instance in self.instances.items():
                        action = self._pending.pop(instance_id, None)
                        if action is not None:
                            instance.submit(action)
                        else:
                            instance.hold()
                    self.world.step()
                    self._publish()
            self._stop.wait(max(0.0, period - (time.monotonic() - started)))

    def _publish(self) -> None:
        with self._lock:
            self._state = build_state_snapshot(
                self.model,
                self.data,
                step=self.world.step_count,
                robot="shared",
                instance_prefixes={
                    instance_id: instance.prefix
                    for instance_id, instance in self.instances.items()
                },
            )

    def _start_camera_thread(self) -> None:
        if self._camera_thread is not None:
            return
        self._camera_thread = threading.Thread(
            target=self._camera_loop,
            name="physai-shared-camera",
            daemon=True,
        )
        self._camera_thread.start()

    def _request_camera_capture(self, *, wait: bool = False) -> None:
        self._camera_ready.clear()
        self._camera_request.set()
        if wait:
            self._camera_ready.wait(timeout=2.0)

    def _camera_loop(self) -> None:
        renderer = mujoco.Renderer(self.model, height=240, width=320)
        next_capture = 0.0
        try:
            while not self._stop.is_set():
                timeout = max(0.0, next_capture - time.monotonic())
                requested = self._camera_request.wait(timeout=timeout)
                self._camera_request.clear()
                if self._stop.is_set():
                    break
                if not requested and time.monotonic() < next_capture:
                    continue
                with self._physics_lock:
                    mujoco.mj_copyData(self._camera_data, self.model, self.data)
                for instance_id, instance in self.instances.items():
                    for name, camera in instance.robot_spec.camera_frames.items():
                        renderer.update_scene(self._camera_data, camera=camera)
                        image = np.asarray(renderer.render(), dtype=np.uint8).copy()
                        with self._lock:
                            self._camera_images[f"{instance_id}:{name}"] = image
                next_capture = time.monotonic() + self._CAMERA_PERIOD
                self._camera_ready.set()
        finally:
            renderer.close()
