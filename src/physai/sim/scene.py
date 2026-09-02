"""Compatibility facade for task-specific MuJoCo scenes."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import mujoco

from .scenes.common import ManipulationSceneConfig
from .scenes.pick_place_minimal import PickPlaceMinimalSceneConfig, build_spec as build_pick_place_spec
from .scenes.sorting_minimal import SortingMinimalSceneConfig, build_spec as build_sorting_spec


@dataclass
class SceneConfig(ManipulationSceneConfig):
    """Legacy scene config that selects a minimal task scene.

    New code should use ``PickPlaceMinimalSceneConfig`` or
    ``SortingMinimalSceneConfig`` directly.
    """

    cube_half: float = 0.014
    cube_pos: tuple[float, float, float] = (0.20, 0.08, 0.036)
    cube_mass: float = 0.03
    cube_rgba: tuple[float, float, float, float] = (0.85, 0.25, 0.2, 1.0)
    num_cubes: int = 1
    sorting_cube_names: tuple[str, ...] = ("cube_red", "cube_blue", "cube_yellow")
    sorting_cube_rgba: tuple[tuple[float, float, float, float], ...] = (
        (0.85, 0.25, 0.2, 1.0),
        (0.2, 0.35, 0.85, 1.0),
        (0.9, 0.8, 0.15, 1.0),
    )

    @property
    def cube_names(self) -> tuple[str, ...]:
        if self.num_cubes == 1:
            return ("cube",)
        return self.sorting_cube_names


def _common_kwargs(cfg: SceneConfig) -> dict:
    return {field.name: getattr(cfg, field.name)
            for field in fields(ManipulationSceneConfig)}


def _pick_place_config(cfg: SceneConfig) -> PickPlaceMinimalSceneConfig:
    return PickPlaceMinimalSceneConfig(
        **_common_kwargs(cfg),
        cube_half=cfg.cube_half,
        cube_pos=cfg.cube_pos,
        cube_mass=cfg.cube_mass,
        cube_rgba=cfg.cube_rgba,
    )


def _sorting_config(cfg: SceneConfig) -> SortingMinimalSceneConfig:
    return SortingMinimalSceneConfig(
        **_common_kwargs(cfg),
        cube_half=cfg.cube_half,
        cube_pos=cfg.cube_pos,
        cube_mass=cfg.cube_mass,
        cube_names=cfg.sorting_cube_names,
        cube_rgba=cfg.sorting_cube_rgba,
    )


def build_spec(
    cfg: SceneConfig | PickPlaceMinimalSceneConfig | SortingMinimalSceneConfig | None = None,
) -> mujoco.MjSpec:
    """Build the selected minimal scene through the legacy config surface."""
    cfg = cfg or SceneConfig()
    if isinstance(cfg, PickPlaceMinimalSceneConfig):
        return build_pick_place_spec(cfg)
    if isinstance(cfg, SortingMinimalSceneConfig):
        return build_sorting_spec(cfg)
    if cfg.num_cubes == 1:
        return build_pick_place_spec(_pick_place_config(cfg))
    if cfg.num_cubes == len(cfg.sorting_cube_names):
        return build_sorting_spec(_sorting_config(cfg))
    raise ValueError(
        f"num_cubes={cfg.num_cubes} not supported — use 1 (default) or "
        f"{len(cfg.sorting_cube_names)} (sorting_cube_names)"
    )


def build_model(
    cfg: SceneConfig | PickPlaceMinimalSceneConfig | SortingMinimalSceneConfig | None = None,
):
    spec = build_spec(cfg)
    return spec.compile(), spec


def export_xml(
    path: Path,
    cfg: SceneConfig | PickPlaceMinimalSceneConfig | SortingMinimalSceneConfig | None = None,
) -> Path:
    """Write a selected minimal scene to disk."""
    spec = build_spec(cfg)
    spec.compile()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(spec.to_xml(), encoding="utf-8")
    return path


if __name__ == "__main__":
    model, _ = build_model()
    print(f"compiled: nq={model.nq} nu={model.nu} nbody={model.nbody} ncam={model.ncam}")
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, index)
             for index in range(model.ncam)]
    print("cameras:", names)
