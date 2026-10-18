"""MuJoCo simulation lifecycle shared by direct and ROS2 adapters."""

from __future__ import annotations

import threading

import mujoco

from ...contracts import CAMERA_HEIGHT, CAMERA_WIDTH
from ..workspace import VIDEO_CAMERA, ViewCamera


class MuJoCoRenderer:
    """Draw a model's cameras with MuJoCo, from a `data` that something keeps current.

    A renderer is any object with `render(camera_name) -> HxWx3 uint8` and
    `close()`; this one is the default, and another simulator's renderer
    (Newton, Isaac) replaces it on `MuJoCoSimulationCore.renderer`. The
    underlying `mujoco.Renderer` belongs to the thread that first rendered.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        width: int,
        height: int,
        view: ViewCamera | None = None,
    ) -> None:
        self.model, self.data = model, data
        self._size = (width, height)
        self._view = view
        self._renderer: mujoco.Renderer | None = None
        self._renderer_thread_id: int | None = None

    def render(self, name: str) -> object:
        thread_id = threading.get_ident()
        if self._renderer is None or self._renderer_thread_id != thread_id:
            if self._renderer is not None:
                self._renderer.close()
            width, height = self._size
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
            self._renderer_thread_id = thread_id
        camera: object = name
        if name == VIDEO_CAMERA and self._view is not None:
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = self._view.lookat
            camera.distance = self._view.distance
            camera.azimuth = self._view.azimuth
            camera.elevation = self._view.elevation
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
            self._renderer_thread_id = None


class MuJoCoSimulationCore:
    """Own model state, stepping, rendering, and simulation time."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        control_hz: float,
        render: bool = False,
        camera_width: int = CAMERA_WIDTH,
        camera_height: int = CAMERA_HEIGHT,
        video_view: ViewCamera | None = None,
    ) -> None:
        if control_hz <= 0:
            raise ValueError("control_hz must be positive")
        self.model = model
        self.data = mujoco.MjData(model)
        self.n_substeps = max(1, round((1.0 / control_hz) / model.opt.timestep))
        self.control_dt = self.n_substeps * model.opt.timestep
        self.step_count = 0
        self._camera_size = (camera_width, camera_height) if render else None
        self.renderer = (
            MuJoCoRenderer(model, self.data, camera_width, camera_height, video_view)
            if render
            else None
        )

    @property
    def render_enabled(self) -> bool:
        """Whether this simulation can render camera frames."""
        return self._camera_size is not None

    @property
    def camera_size(self) -> tuple[int, int] | None:
        """(width, height) of rendered frames, or None when rendering is off."""
        return self._camera_size

    def reset_simulation(self) -> None:
        """Reset simulator state before an adapter applies its initial state."""
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0

    def step_simulation(self) -> None:
        """Advance the simulator by one control period."""
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1

    def render_camera(self, name: str) -> object:
        if self.renderer is None:
            raise RuntimeError("simulation constructed with render=False")
        return self.renderer.render(name)

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
