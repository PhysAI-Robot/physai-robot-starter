"""Color-sorting task: move the cube matching backend.target_color onto the pad.

Same success geometry as PickPlaceTask, but the target cube is picked by
color out of several on the table instead of being the only cube present.
backend.cube_pos already resolves to the target-colored cube's position (see
SO101PickPlaceEnv.cube_pos), so evaluate() only has to expose which color was
asked for.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import Task


class SortingTask(Task):
    name = "sorting"

    def __init__(self, success_xy_tol: float = 0.04) -> None:
        self.success_xy_tol = success_xy_tol

    def evaluate(self, backend: Any) -> dict:
        cube = np.asarray(backend.cube_pos)
        target = np.asarray(backend.target_pos)
        ee = np.asarray(backend.ee_pos)
        table_top = backend.table_top
        cube_half = backend.cube_half
        return {
            "target_color": backend.target_color,
            "cube_pos": cube,
            "cube_positions": backend.cube_positions,
            "target_pos": target,
            "ee_pos": ee,
            "dist_ee_cube": float(np.linalg.norm(ee - cube)),
            "dist_cube_target": float(np.linalg.norm(cube[:2] - target[:2])),
            "cube_lifted": bool(cube[2] > table_top + 2.5 * cube_half),
            "cube_dropped": bool(cube[2] < table_top - 0.05),
            "at_target": bool(
                np.linalg.norm(cube[:2] - target[:2]) < self.success_xy_tol
                and abs(cube[2] - (table_top + cube_half)) < 0.02
            ),
        }

    def reward(self, backend: Any, info: dict | None = None) -> float:
        info = info or self.evaluate(backend)
        value = -0.02 * info["dist_ee_cube"] - 0.05 * info["dist_cube_target"]
        if info["cube_lifted"]:
            value += 0.2
        if info["at_target"]:
            value += 1.0
        if info["cube_dropped"]:
            value -= 1.0
        return float(value)
