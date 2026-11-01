"""Cube-stacking task: pile the cubes on the pad in a given order, bottom first."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .base import Task
from .single_cube_place import DEFAULT_SUCCESS_XY_TOL

# How far a cube's height may be from its level, in metres.
DEFAULT_LEVEL_Z_TOL = 0.008
DEFAULT_ORDER = ("red", "blue")


class StackCubesBackend(Protocol):
    """Geometry state required by stacking evaluation."""

    cube_positions: dict[str, np.ndarray]
    target_pos: np.ndarray
    ee_pos: np.ndarray
    table_top: float
    cube_half: float


class StackCubesTask(Task):
    """The cubes named in `order` stacked on the pad, `order[0]` at the bottom.

    Level k is right when that cube's centre is over the pad (`success_xy_tol`) and at the
    height of the k-th cube of a tower (`level_z_tol`). `at_target` needs every level right,
    which the runtime holds for several steps before it counts as success.
    """

    name = "stack_cubes"
    required_capabilities = ("arm_kinematics", "gripper")
    required_action_modes = ("joint_position",)

    def __init__(
        self,
        order: tuple[str, ...] = DEFAULT_ORDER,
        success_xy_tol: float = DEFAULT_SUCCESS_XY_TOL,
        level_z_tol: float = DEFAULT_LEVEL_Z_TOL,
    ) -> None:
        if len(set(order)) != len(order) or not order:
            raise ValueError("stack order must name each cube once")
        self.order = tuple(order)
        self.success_xy_tol = success_xy_tol
        self.level_z_tol = level_z_tol

    def evaluate(self, backend: StackCubesBackend) -> dict:
        cubes = {name: np.asarray(backend.cube_positions[name]) for name in self.order}
        pad = np.asarray(backend.target_pos)
        rest_z = backend.table_top + backend.cube_half
        right = []
        for level, name in enumerate(self.order):
            cube = cubes[name]
            right.append(
                bool(
                    np.linalg.norm(cube[:2] - pad[:2]) < self.success_xy_tol
                    and abs(cube[2] - (rest_z + 2 * backend.cube_half * level))
                    < self.level_z_tol
                )
            )
        stacked = 0
        for ok in right:  # levels count from the bottom, so a gap ends the count
            if not ok:
                break
            stacked += 1
        return {
            "cube_positions": cubes,
            "target_pos": pad,
            "ee_pos": np.asarray(backend.ee_pos),
            "cubes_stacked": stacked,
            "cube_dropped": bool(
                any(c[2] < backend.table_top - 0.05 for c in cubes.values())
            ),
            "at_target": stacked == len(self.order),
        }

    def reward(self, backend: StackCubesBackend, info: dict | None = None) -> float:
        info = info or self.evaluate(backend)
        pad = info["target_pos"]
        away = sum(
            float(np.linalg.norm(c[:2] - pad[:2]))
            for c in info["cube_positions"].values()
        )
        value = -0.05 * away + 0.5 * info["cubes_stacked"]
        if info["at_target"]:
            value += 1.0
        if info["cube_dropped"]:
            value -= 1.0
        return float(value)

    def terminated(self, backend: StackCubesBackend, info: dict) -> bool:
        return bool(info.get("success") or info.get("cube_dropped"))
