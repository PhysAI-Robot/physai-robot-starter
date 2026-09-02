"""Minimal three-color sorting MuJoCo scene."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco

from .common import ManipulationSceneConfig, add_cube, build_manipulation_spec


@dataclass
class SortingMinimalSceneConfig(ManipulationSceneConfig):
    cube_half: float = 0.014
    cube_pos: tuple[float, float, float] = (0.20, 0.08, 0.036)
    cube_mass: float = 0.03
    cube_names: tuple[str, ...] = ("cube_red", "cube_blue", "cube_yellow")
    cube_rgba: tuple[tuple[float, float, float, float], ...] = (
        (0.85, 0.25, 0.2, 1.0),
        (0.2, 0.35, 0.85, 1.0),
        (0.9, 0.8, 0.15, 1.0),
    )


def build_spec(cfg: SortingMinimalSceneConfig | None = None) -> mujoco.MjSpec:
    cfg = cfg or SortingMinimalSceneConfig()
    if len(cfg.cube_names) != len(cfg.cube_rgba):
        raise ValueError("sorting cube names and colors must have the same length")
    spec = build_manipulation_spec(cfg)
    for index, (name, rgba) in enumerate(zip(cfg.cube_names, cfg.cube_rgba)):
        position = (cfg.cube_pos[0], cfg.cube_pos[1] + 0.06 * index, cfg.cube_pos[2])
        add_cube(spec, cfg, name, position, rgba, cfg.cube_half, cfg.cube_mass)
    return spec


def build_model(cfg: SortingMinimalSceneConfig | None = None):
    spec = build_spec(cfg)
    return spec.compile(), spec
