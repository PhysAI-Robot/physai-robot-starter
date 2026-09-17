"""MuJoCo simulation lifecycle shared by direct and ROS2 adapters."""

from __future__ import annotations

import threading

import mujoco


class MuJoCoSimulationCore:
    """Own model state, stepping, rendering, and simulation time."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        control_hz: float,
        render: bool = False,
        camera_width: int = 640,
        camera_height: int = 480,
    ) -> None:
        if control_hz <= 0:
            raise ValueError("control_hz must be positive")
        self.model = model
        self.data = mujoco.MjData(model)
        self.n_substeps = max(1, round((1.0 / control_hz) / model.opt.timestep))
        self.control_dt = self.n_substeps * model.opt.timestep
        self.step_count = 0
        self._renderer: mujoco.Renderer | None = None
        self._renderer_thread_id: int | None = None
        if render:
            self._camera_width = camera_width
            self._camera_height = camera_height

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
        if not hasattr(self, "_camera_width"):
            raise RuntimeError("simulation constructed with render=False")
        thread_id = threading.get_ident()
        if self._renderer is None or self._renderer_thread_id != thread_id:
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(
                self.model,
                height=self._camera_height,
                width=self._camera_width,
            )
            self._renderer_thread_id = thread_id
        self._renderer.update_scene(self.data, camera=name)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
            self._renderer_thread_id = None
