"""SO-101 spawn defaults; the layouts themselves are shared (`..layout`)."""

from __future__ import annotations

from ..layout import (
    XY,
    LayoutConfig,
    ObjectLayout,
    SingleCubeLayout,
    SortingLayout,
    create_layout,
    draw_cube_xy,
    draw_target_xy,
    draw_xy,
)

# Where a single cube may start, when its position is randomized per episode.
# Shared so every simulator's env config defaults to the same layout.
DEFAULT_CUBE_X_RANGE: tuple[float, float] = (0.20, 0.24)
DEFAULT_CUBE_Y_RANGE: tuple[float, float] = (0.05, 0.13)
# Where the target pad may go when it is randomized.
DEFAULT_TARGET_X_RANGE: tuple[float, float] = (0.16, 0.26)
DEFAULT_TARGET_Y_RANGE: tuple[float, float] = (-0.13, -0.04)

__all__ = [
    "DEFAULT_CUBE_X_RANGE",
    "DEFAULT_CUBE_Y_RANGE",
    "DEFAULT_TARGET_X_RANGE",
    "DEFAULT_TARGET_Y_RANGE",
    "XY",
    "LayoutConfig",
    "ObjectLayout",
    "SingleCubeLayout",
    "SortingLayout",
    "create_layout",
    "draw_cube_xy",
    "draw_target_xy",
    "draw_xy",
]
