"""MuJoCo simulation lifecycle shared by direct and ROS2 adapters."""

from __future__ import annotations

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
        if render:
            self._renderer = mujoco.Renderer(
                model,
                height=camera_height,
                width=camera_width,
            )

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
        if self._renderer is None:
            raise RuntimeError("simulation constructed with render=False")
        self._renderer.update_scene(self.data, camera=name)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
