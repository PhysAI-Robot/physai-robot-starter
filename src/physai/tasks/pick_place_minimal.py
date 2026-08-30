"""Cube pick-and-place task logic, independent of the SO-101 implementation."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .base import Task, TaskBackend


class PickPlaceBackend(TaskBackend, Protocol):
    """Minimal geometry state required by pick-and-place evaluation."""

    cube_pos: np.ndarray
    target_pos: np.ndarray
    ee_pos: np.ndarray
    table_top: float
    cube_half: float


class PickPlaceTask(Task):
    name = "pick_place"
    required_capabilities = ("arm_kinematics", "gripper")
    required_action_modes = ("joint_position",)

    def __init__(self, success_xy_tol: float = 0.04) -> None:
        self.success_xy_tol = success_xy_tol

    def evaluate(self, backend: PickPlaceBackend) -> dict:
        cube = np.asarray(backend.cube_pos)
        target = np.asarray(backend.target_pos)
        ee = np.asarray(backend.ee_pos)
        table_top = backend.table_top
        cube_half = backend.cube_half
        return {
            "cube_pos": cube,
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

    def reward(self, backend: PickPlaceBackend, info: dict | None = None) -> float:
        info = info or self.evaluate(backend)
        value = -0.02 * info["dist_ee_cube"] - 0.05 * info["dist_cube_target"]
        if info["cube_lifted"]:
            value += 0.2
        if info["at_target"]:
            value += 1.0
        if info["cube_dropped"]:
            value -= 1.0
        return float(value)

    def terminated(self, backend: PickPlaceBackend, info: dict) -> bool:
        return bool(info.get("success") or info.get("cube_dropped"))
