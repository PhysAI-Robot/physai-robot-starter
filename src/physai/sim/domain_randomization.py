"""Seeded MuJoCo domain randomization with episode metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np


@dataclass(frozen=True)
class DomainRandomizationConfig:
    """Safe ranges for the supported MuJoCo randomization parameters."""

    enabled: bool = False
    friction_scale: tuple[float, float] = (0.9, 1.1)
    mass_scale: tuple[float, float] = (0.95, 1.05)
    lighting_scale: tuple[float, float] = (0.9, 1.1)
    camera_position_jitter: float = 0.0
    clutter_x_range: tuple[float, float] = (0.14, 0.28)
    clutter_y_range: tuple[float, float] = (-0.16, 0.16)
    clutter_clearance: float = 0.05

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("domain_randomization.enabled must be a boolean")
        for name in ("friction_scale", "mass_scale", "lighting_scale"):
            bounds = getattr(self, name)
            if len(bounds) != 2 or not all(np.isfinite(bounds)):
                raise ValueError(f"{name} must contain two finite values")
            if bounds[0] <= 0 or bounds[0] > bounds[1]:
                raise ValueError(f"{name} must satisfy 0 < low <= high")
        for name in ("clutter_x_range", "clutter_y_range"):
            bounds = getattr(self, name)
            if len(bounds) != 2 or not all(np.isfinite(bounds)):
                raise ValueError(f"{name} must contain two finite values")
            if bounds[0] > bounds[1]:
                raise ValueError(f"{name} must satisfy low <= high")
        if not np.isfinite(self.camera_position_jitter) or self.camera_position_jitter < 0:
            raise ValueError("camera_position_jitter must be finite and non-negative")
        if not np.isfinite(self.clutter_clearance) or self.clutter_clearance < 0:
            raise ValueError("clutter_clearance must be finite and non-negative")


@dataclass(frozen=True)
class RandomizationMetadata:
    """JSON-compatible parameters sampled for one episode."""

    enabled: bool
    seed: int | None
    friction_scale: float
    mass_scale: float
    lighting_scale: float
    camera_position_offset: dict[str, tuple[float, float, float]]
    clutter_position: dict[str, tuple[float, float]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "seed": self.seed,
            "friction_scale": self.friction_scale,
            "mass_scale": self.mass_scale,
            "lighting_scale": self.lighting_scale,
            "camera_position_offset": {
                name: list(offset)
                for name, offset in self.camera_position_offset.items()
            },
            "clutter_position": {
                name: list(position)
                for name, position in self.clutter_position.items()
            },
        }


class DomainRandomizationEngine:
    """Restore and randomize mutable model parameters per episode."""

    def __init__(
        self,
        model: mujoco.MjModel,
        config: DomainRandomizationConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or DomainRandomizationConfig()
        self._base_geom_friction = model.geom_friction.copy()
        self._base_body_mass = model.body_mass.copy()
        self._base_light_diffuse = model.light_diffuse.copy()
        self._base_cam_pos = model.cam_pos.copy()
        self._base_geom_pos = model.geom_pos.copy()

    def _restore(self) -> None:
        self.model.geom_friction[:] = self._base_geom_friction
        self.model.body_mass[:] = self._base_body_mass
        self.model.light_diffuse[:] = self._base_light_diffuse
        self.model.cam_pos[:] = self._base_cam_pos
        self.model.geom_pos[:] = self._base_geom_pos

    def apply(
        self,
        rng: np.random.Generator,
        *,
        seed: int | None = None,
        protected_xy: tuple[tuple[float, float], ...] = (),
    ) -> RandomizationMetadata:
        """Apply one seeded sample and return the recorded parameters."""
        self._restore()
        if not self.config.enabled:
            return RandomizationMetadata(
                enabled=False,
                seed=seed,
                friction_scale=1.0,
                mass_scale=1.0,
                lighting_scale=1.0,
                camera_position_offset={},
                clutter_position={},
            )

        friction_scale = float(rng.uniform(*self.config.friction_scale))
        mass_scale = float(rng.uniform(*self.config.mass_scale))
        lighting_scale = float(rng.uniform(*self.config.lighting_scale))
        self.model.geom_friction[:] *= friction_scale
        self.model.body_mass[1:] *= mass_scale
        self.model.light_diffuse[:] *= lighting_scale

        offsets: dict[str, tuple[float, float, float]] = {}
        jitter = self.config.camera_position_jitter
        for camera_id in range(self.model.ncam):
            offset = rng.uniform(-jitter, jitter, size=3)
            self.model.cam_pos[camera_id] += offset
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id
            ) or f"camera_{camera_id}"
            offsets[name] = tuple(float(value) for value in offset)
        clutter_positions: dict[str, tuple[float, float]] = {}
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
            )
            if not name or not name.startswith("physai_clutter_"):
                continue
            x, y = 0.0, 0.0
            for _ in range(100):
                candidate = np.array([
                    rng.uniform(*self.config.clutter_x_range),
                    rng.uniform(*self.config.clutter_y_range),
                ])
                if all(
                    np.linalg.norm(candidate - np.asarray(point))
                    >= self.config.clutter_clearance
                    for point in protected_xy
                ):
                    x, y = (float(candidate[0]), float(candidate[1]))
                    break
            else:
                raise ValueError(
                    "unable to place clutter outside protected task positions"
                )
            self.model.geom_pos[geom_id, :2] = (x, y)
            clutter_positions[name] = (x, y)
        return RandomizationMetadata(
            enabled=True,
            seed=seed,
            friction_scale=friction_scale,
            mass_scale=mass_scale,
            lighting_scale=lighting_scale,
            camera_position_offset=offsets,
            clutter_position=clutter_positions,
        )


__all__ = [
    "DomainRandomizationConfig",
    "DomainRandomizationEngine",
    "RandomizationMetadata",
]
