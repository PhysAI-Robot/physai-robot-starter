"""Minimal three-color sorting MuJoCo scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ...workspace import CubeSpec
from .common import ManipulationSceneConfig


@dataclass
class SortingMinimalSceneConfig(ManipulationSceneConfig):
    layout_kind: ClassVar[str] = "sorting"
    cube_half: float = 0.014
    cube_pos: tuple[float, float, float] = (0.20, 0.08, 0.036)
    cube_mass: float = 0.03
    cube_names: tuple[str, ...] = ("cube_red", "cube_blue", "cube_yellow")
    cube_rgba: tuple[tuple[float, float, float, float], ...] = (
        (0.85, 0.25, 0.2, 1.0),
        (0.2, 0.35, 0.85, 1.0),
        (0.9, 0.8, 0.15, 1.0),
    )

    def cubes(self) -> tuple[CubeSpec, ...]:
        if len(self.cube_names) != len(self.cube_rgba):
            raise ValueError("sorting cube names and colors must have the same length")
        return tuple(
            CubeSpec(
                name,
                (
                    self.cube_pos[0],
                    self.cube_pos[1] + 0.06 * index,
                    self.cube_pos[2],
                ),
                self.cube_half,
                self.cube_mass,
                rgba,
            )
            for index, (name, rgba) in enumerate(zip(self.cube_names, self.cube_rgba))
        )
