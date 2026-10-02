"""Isaac Sim simulation lifecycle: the Isaac analogue of `sim.mujoco.core
.MuJoCoSimulationCore` — one process-wide `SimulationApp`, stage and
physics-scene bootstrap, stepping, and camera rendering.

Only one `SimulationApp` may exist per process (a hard Isaac Sim
constraint), so this module owns creating it lazily on first use and never
recreates it; every Isaac-backed robot in the same process shares it. This
mirrors `sim.mujoco.core` owning MuJoCo's model/data lifecycle, and is the only
place besides `robots.so101.isaac_env` that may import `isaacsim`/`omni`/
`pxr` (see the import-linter contract in `pyproject.toml`).
"""

from __future__ import annotations

from typing import Any

from ...contracts import CAMERA_HEIGHT, CAMERA_WIDTH

_APP: Any | None = None


def ensure_simulation_app(*, headless: bool = True) -> Any:
    """Return the process-wide `SimulationApp`, creating it on first call.

    A second call — even with a different `headless` value — returns the
    existing app unchanged; Isaac Sim cannot run two in one process.
    """
    global _APP
    if _APP is None:
        from isaacsim import SimulationApp

        _APP = SimulationApp({"headless": headless})
    return _APP


def close_simulation_app() -> None:
    """Shut down the process-wide `SimulationApp`, if one was created.

    Irreversible for the rest of the process: Isaac Sim cannot be
    re-launched once closed. Call this only at process exit (e.g. test
    session teardown), never between episodes.
    """
    global _APP
    if _APP is not None:
        _APP.close()
        _APP = None


class IsaacSimulationCore:
    """Own stage state, physics stepping, rendering, and simulation time.

    Constructed after `ensure_simulation_app()` and after the USD stage is
    open with a robot already imported; this class only owns the generic
    stepping/render loop, the way `MuJoCoSimulationCore` does not build a
    scene either.
    """

    def __init__(
        self,
        *,
        control_hz: float,
        physics_hz: float = 60.0,
        render: bool = False,
        camera_width: int = CAMERA_WIDTH,
        camera_height: int = CAMERA_HEIGHT,
    ) -> None:
        if control_hz <= 0 or physics_hz <= 0:
            raise ValueError("control_hz and physics_hz must be positive")
        from isaacsim.core.simulation_manager import SimulationManager

        self._sim = SimulationManager
        self._sim.set_physics_dt(1.0 / physics_hz)
        self.n_substeps = max(1, round(physics_hz / control_hz))
        self.control_dt = self.n_substeps / physics_hz
        self.step_count = 0
        self._camera_size = (camera_width, camera_height) if render else None
        self._render_products: dict[str, Any] = {}

    @property
    def render_enabled(self) -> bool:
        return self._camera_size is not None

    @property
    def camera_size(self) -> tuple[int, int] | None:
        return self._camera_size

    def reset_simulation(self) -> None:
        """Reset physics state before an adapter applies its initial state."""
        self._sim.initialize_physics()
        self.step_count = 0

    def step_simulation(self) -> None:
        """Advance the simulator by one control period."""
        for _ in range(self.n_substeps):
            self._sim.step()
        self.step_count += 1

    def render_camera(self, prim_path: str) -> Any:
        """Render one RGB frame from the USD camera at `prim_path`.

        Uses a Replicator render product + RGB annotator per camera,
        created lazily and reused: creating one per call is expensive and
        each camera's pose is static relative to its own render product.
        """
        if self._camera_size is None:
            raise RuntimeError("simulation constructed with render=False")
        import numpy as np
        import omni.replicator.core as rep

        annotator = self._render_products.get(prim_path)
        is_new = annotator is None
        if is_new:
            width, height = self._camera_size
            render_product = rep.create.render_product(
                prim_path, resolution=(width, height)
            )
            annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            annotator.attach(render_product)
            self._render_products[prim_path] = annotator
        rep.orchestrator.step(rt_subframes=1)
        frame = np.asarray(annotator.get_data())
        # A render product's first frame, right after attach(), can come
        # back empty (1-D) when this process's Replicator orchestrator is
        # already running from an earlier env instance's camera(s) -- a
        # fresh-in-process orchestrator doesn't need this, but stepping a
        # couple more times costs little and makes a later env's first
        # render call as reliable as its own second one.
        retries = 5 if is_new else 0
        while frame.ndim != 3 and retries > 0:
            rep.orchestrator.step(rt_subframes=1)
            frame = np.asarray(annotator.get_data())
            retries -= 1
        return frame[:, :, :3].copy()

    def close(self) -> None:
        self._render_products.clear()
