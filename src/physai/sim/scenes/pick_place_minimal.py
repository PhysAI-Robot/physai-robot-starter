"""Minimal single-cube pick-and-place MuJoCo scene."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco

from .common import ManipulationSceneConfig, add_cube, build_manipulation_spec


@dataclass
class PickPlaceMinimalSceneConfig(ManipulationSceneConfig):
    cube_names: tuple[str, ...] = ("cube",)
    cube_half: float = 0.014
    cube_pos: tuple[float, float, float] = (0.20, 0.08, 0.036)
    cube_mass: float = 0.03
    cube_rgba: tuple[float, float, float, float] = (0.85, 0.25, 0.2, 1.0)


def build_spec(cfg: PickPlaceMinimalSceneConfig | None = None) -> mujoco.MjSpec:
    cfg = cfg or PickPlaceMinimalSceneConfig()
    spec = build_manipulation_spec(cfg)
    add_cube(spec, cfg, "cube", cfg.cube_pos, cfg.cube_rgba,
             cfg.cube_half, cfg.cube_mass)
    return spec


def build_model(cfg: PickPlaceMinimalSceneConfig | None = None):
    spec = build_spec(cfg)
    return spec.compile(), spec
