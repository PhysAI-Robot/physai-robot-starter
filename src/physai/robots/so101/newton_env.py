"""SO-101 on Newton physics: the MuJoCo env, stepped by Newton's solver."""

from __future__ import annotations

from dataclasses import replace
from functools import cached_property
from pathlib import Path

from ...sim.newton.core import NewtonPhysics
from ..base import RobotSpec
from .mujoco_env import EnvConfig, SO101Env


class SO101NewtonEnv(NewtonPhysics, SO101Env):
    def __init__(self, cfg: EnvConfig | None = None) -> None:
        super().__init__(cfg)
        self._start_newton(Path(self.cfg.scene.robot_xml).parent)

    @cached_property
    def robot_spec(self) -> RobotSpec:
        spec = SO101Env.robot_spec.func(self)
        return replace(spec, metadata={**spec.metadata, "simulator": "newton"})
