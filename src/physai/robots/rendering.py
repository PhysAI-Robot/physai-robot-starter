"""Pair a simulator with another simulator's camera renderer.

Every manipulator env keeps its scene in a MuJoCo `model`/`data` pair: the
simulation itself for MuJoCo and Newton, the display mirror for Isaac. A
renderer is anything with `render(camera) -> HxWx3 uint8` and `close()` that
draws from that pair, so `--render` is independent of `--sim`.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import mujoco

from ..sim.mujoco.core import MuJoCoRenderer

ENGINES = ("mujoco", "isaac", "newton")


class IsaacRenderer:
    """Draw another simulator's state with a render-only Isaac env."""

    def __init__(self, env: Any, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.env, self.model, self.data = env, model, data

    def render(self, name: str) -> Any:
        self.env.show_state(self.model, self.data)
        return self.env.render_camera(name)

    def close(self) -> None:
        self.env.close()


def start_isaac_first(simulator: str, renderer: str | None) -> None:
    """Start Isaac Sim before anything loads warp.

    Kit brings its own warp; an Isaac app started after Newton has initialized
    the pip one comes up without `omni.usd`.
    """
    if "isaac" in (simulator, renderer):
        from ..sim.isaac.core import ensure_simulation_app

        ensure_simulation_app()


def _reject_unrendered_looks(env: Any, simulator: str, renderer: str) -> None:
    """Visual randomization lives in one simulator's scene; another would not draw it."""
    cfg = env.cfg
    if simulator == "isaac":
        looks = cfg.lighting_scale != 1.0 or cfg.camera_position_jitter != 0.0
    else:
        looks = cfg.domain_randomization.enabled
    if looks:
        raise ValueError(
            f"--sim {simulator} --render {renderer} cannot draw the configured "
            "randomization (lighting, camera jitter, domain_randomization); "
            f"render with {simulator}"
        )


def attach_renderer(
    env: Any,
    simulator: str,
    renderer: str | None,
    make_isaac_env: Callable[[Any], Any],
) -> None:
    """Replace `env.renderer` with `renderer`'s, drawing from the env's model and data.

    `make_isaac_env(scene)` builds a render-only Isaac env for the Isaac case.
    """
    if renderer in (None, simulator):
        return
    if renderer not in ENGINES:
        raise ValueError(
            f"unknown renderer {renderer!r}; available: {', '.join(ENGINES)}"
        )
    scene = env.cfg.scene
    if scene is None:
        raise ValueError(f"--render {renderer} needs a scene to draw")
    _reject_unrendered_looks(env, simulator, renderer)
    width, height = scene.camera_width, scene.camera_height
    if renderer == "mujoco":
        new = MuJoCoRenderer(env.model, env.data, width, height, scene.video_view)
    elif renderer == "newton":
        from ..sim.newton.render import NewtonRenderer

        spec = getattr(env, "spec", None)
        if spec is None:  # the Isaac env keeps no spec
            spec = scene.build_spec()
            spec.compile()
        new = NewtonRenderer(
            spec,
            env.model,
            env.data,
            robot_dir=Path(scene.robot_xml).parent,
            width=width,
            height=height,
            view=scene.video_view,
        )
    else:
        new = IsaacRenderer(make_isaac_env(scene), env.model, env.data)
    if env.renderer is not None:
        env.renderer.close()
    env.renderer = new
