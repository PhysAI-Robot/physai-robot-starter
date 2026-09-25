"""Where a scene's objects start each episode, and how to read them back.

`SO101Env.reset` places the arm; the object layout of the scene it was built
for is a separate strategy chosen by the scene's `layout_kind`. Adding a scene
with a new arrangement (another object, several targets) means adding a layout
here and naming it on the scene, not editing the environment. The order of
random-number draws is part of the contract: a seed must keep producing the
same scene (see `tests/acceptance/so101/test_object_layout.py`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from ...sim.scenes import ManipulationSceneConfig
    from .env import EnvConfig

XY = tuple[float, float]
_UPRIGHT = [1, 0, 0, 0]


class ObjectLayout(ABC):
    """Places a scene's cubes and target pad, and reports the cubes' state."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self._model = model
        self._target_site = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "target_site"
        )

    @property
    def target_color(self) -> str | None:
        """The color the current episode asks for, for layouts that have one."""
        return None

    @abstractmethod
    def cube_positions(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        """Every colored cube's position; empty for a single-cube layout."""

    @abstractmethod
    def cube_pos(self, data: mujoco.MjData) -> np.ndarray:
        """The position of the cube the task is asking for."""

    @abstractmethod
    def cube_geom_id(self) -> int:
        """The collision geom of the cube the task is asking for."""

    @abstractmethod
    def _place_cubes(
        self, data: mujoco.MjData, rng: np.random.Generator, cfg: EnvConfig
    ) -> None: ...

    @abstractmethod
    def _cube_xy(self, data: mujoco.MjData) -> list[XY]: ...

    def reset(
        self, data: mujoco.MjData, rng: np.random.Generator, cfg: EnvConfig
    ) -> tuple[XY, ...]:
        """Place cubes, then the target pad; return the xy that clutter must avoid."""
        self._place_cubes(data, rng, cfg)
        if cfg.randomize_target:
            self._randomize_target(rng, cfg)
        target_xy = tuple(float(v) for v in self._model.site_pos[self._target_site][:2])
        return (target_xy, *self._cube_xy(data))

    def _randomize_target(self, rng: np.random.Generator, cfg: EnvConfig) -> None:
        target_pos = self._model.site_pos[self._target_site].copy()
        target_pos[0] = rng.uniform(*cfg.target_x_range)
        target_pos[1] = rng.uniform(*cfg.target_y_range)
        self._model.site_pos[self._target_site] = target_pos
        pad = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, "target_pad")
        self._model.geom_pos[pad] = target_pos


class SingleCubeLayout(ObjectLayout):
    """One cube, randomized over a rectangle."""

    def __init__(self, model: mujoco.MjModel, scene: ManipulationSceneConfig) -> None:
        super().__init__(model)
        self._scene = scene
        self._body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        self._qadr = int(model.jnt_qposadr[joint])

    def cube_positions(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        return {}

    def cube_pos(self, data: mujoco.MjData) -> np.ndarray:
        return data.xpos[self._body].copy()

    def cube_geom_id(self) -> int:
        return mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")

    def _place_cubes(self, data, rng, cfg) -> None:
        cube_pos = np.array(self._scene.cube_pos, dtype=np.float64)
        if cfg.randomize_cube:
            cube_pos[0] = rng.uniform(*cfg.cube_x_range)
            cube_pos[1] = rng.uniform(*cfg.cube_y_range)
        data.qpos[self._qadr : self._qadr + 3] = cube_pos
        data.qpos[self._qadr + 3 : self._qadr + 7] = _UPRIGHT

    def _cube_xy(self, data) -> list[XY]:
        return [tuple(float(v) for v in data.qpos[self._qadr : self._qadr + 2])]


class SortingLayout(ObjectLayout):
    """Colored cubes in shuffled lateral bands; one color is asked for per episode."""

    # The lateral (y) bands the cubes are shuffled into, 6 cm apart. Three
    # bands, so a scene with more cubes than this needs its own layout.
    Y_BANDS = ((0.00, 0.02), (0.06, 0.08), (0.12, 0.14))

    def __init__(self, model: mujoco.MjModel, scene: ManipulationSceneConfig) -> None:
        super().__init__(model)
        self._scene = scene
        self._cubes: dict[str, tuple[int, int]] = {}
        for name in scene.cube_names:
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_free")
            self._cubes[name.removeprefix("cube_")] = (
                body,
                int(model.jnt_qposadr[joint]),
            )
        self._target_color: str | None = None

    @property
    def target_color(self) -> str | None:
        return self._target_color

    def cube_positions(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        return {
            color: data.xpos[body].copy() for color, (body, _) in self._cubes.items()
        }

    def cube_pos(self, data: mujoco.MjData) -> np.ndarray:
        body, _ = self._cubes[self._target_color]
        return data.xpos[body].copy()

    def cube_geom_id(self) -> int:
        return mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_GEOM, f"cube_{self._target_color}_geom"
        )

    def _place_cubes(self, data, rng, cfg) -> None:
        base_z = self._scene.cube_pos[2]
        colors = list(self._cubes)
        rng.shuffle(colors)
        for color, (y_lo, y_hi) in zip(colors, self.Y_BANDS):
            _, qadr = self._cubes[color]
            position = np.array([0.0, 0.0, base_z], dtype=np.float64)
            if cfg.randomize_cube:
                position[0] = rng.uniform(*cfg.cube_x_range)
                position[1] = rng.uniform(y_lo, y_hi)
            else:
                position[0] = self._scene.cube_pos[0]
                position[1] = (y_lo + y_hi) / 2
            data.qpos[qadr : qadr + 3] = position
            data.qpos[qadr + 3 : qadr + 7] = _UPRIGHT
        self._target_color = rng.choice(list(self._cubes))

    def _cube_xy(self, data) -> list[XY]:
        return [
            tuple(float(v) for v in data.qpos[qadr : qadr + 2])
            for _, qadr in self._cubes.values()
        ]


_LAYOUTS = {"single_cube": SingleCubeLayout, "sorting": SortingLayout}


def create_layout(
    model: mujoco.MjModel, scene: ManipulationSceneConfig
) -> ObjectLayout:
    """The layout a scene names through its `layout_kind`."""
    try:
        layout = _LAYOUTS[scene.layout_kind]
    except KeyError:
        raise ValueError(
            f"scene {type(scene).__name__} has layout {scene.layout_kind!r}; "
            f"the SO-101 supports: {', '.join(sorted(_LAYOUTS))}"
        ) from None
    return layout(model, scene)


__all__ = [
    "ObjectLayout",
    "SingleCubeLayout",
    "SortingLayout",
    "create_layout",
]
