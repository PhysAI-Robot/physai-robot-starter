"""Color-sorting task: move the cube matching backend.target_color onto the pad.

Same success geometry and reward as PickPlaceTask, but the target cube is
picked by color out of several on the table instead of being the only cube
present. ``backend.cube_pos`` already resolves to the target-colored cube's
position, so evaluate() only has to expose which color was asked for.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .pick_place_minimal import PickPlaceBackend, PickPlaceTask


class SortingBackend(PickPlaceBackend, Protocol):
    """Additional state required by color-sorting evaluation."""

    target_color: str
    cube_positions: dict[str, np.ndarray]


class SortingTask(PickPlaceTask):
    name = "sorting"

    def evaluate(self, backend: SortingBackend) -> dict:
        info = super().evaluate(backend)
        info["target_color"] = backend.target_color
        info["cube_positions"] = backend.cube_positions
        return info
