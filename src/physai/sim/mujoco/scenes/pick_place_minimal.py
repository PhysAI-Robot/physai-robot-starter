"""Minimal single-cube pick-and-place MuJoCo scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import mujoco

from .common import ManipulationSceneConfig, add_cube, build_manipulation_spec


@dataclass
class PickPlaceMinimalSceneConfig(ManipulationSceneConfig):
    layout_kind: ClassVar[str] = "single_cube"
    cube_names: tuple[str, ...] = ("cube",)
    cube_half: float = 0.014
    cube_pos: tuple[float, float, float] = (0.20, 0.08, 0.036)
    cube_mass: float = 0.03
    cube_rgba: tuple[float, float, float, float] = (0.85, 0.25, 0.2, 1.0)

    def build_spec(self) -> mujoco.MjSpec:
        spec = build_manipulation_spec(self)
        add_cube(
            spec,
            self,
            "cube",
            self.cube_pos,
            self.cube_rgba,
            self.cube_half,
            self.cube_mass,
        )
        return spec
