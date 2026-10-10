"""Seeded MuJoCo domain randomization with episode metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    # True: `camera_calibration` reports the shifted pose, so a policy knows
    # where its camera is (a re-calibrated rig). False: it keeps reporting the
    # nominal pose while the camera has moved (a bumped or mis-mounted camera),
    # which is what a robustness-to-camera-shift measurement needs.
    camera_shift_calibrated: bool = True
    clutter_x_range: tuple[float, float] = (0.14, 0.28)
    clutter_y_range: tuple[float, float] = (-0.16, 0.16)
    clutter_clearance: float = 0.05
    # Appearance: per-channel colour multipliers (low, high) sampled each episode for the
    # robot's materials, the table top and the floor, and the floor's tile repeat. The
    # default (1, 1) draws nothing, so existing seeds keep their samples. The cube and the
    # target keep their colours: the task is defined by them.
    arm_tint: tuple[float, float] = (1.0, 1.0)
    table_tint: tuple[float, float] = (1.0, 1.0)
    floor_tint: tuple[float, float] = (1.0, 1.0)
    floor_tile_repeat: tuple[float, float] = (6.0, 6.0)

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError(  # noqa: TRY004
                "domain_randomization.enabled must be a boolean"
            )
        if not isinstance(self.camera_shift_calibrated, bool):
            raise ValueError(  # noqa: TRY004
                "domain_randomization.camera_shift_calibrated must be a boolean"
            )
        for name in ("friction_scale", "mass_scale", "lighting_scale"):
            bounds = getattr(self, name)
            if len(bounds) != 2 or not all(np.isfinite(bounds)):
                raise ValueError(f"{name} must contain two finite values")
            if bounds[0] <= 0 or bounds[0] > bounds[1]:
                raise ValueError(f"{name} must satisfy 0 < low <= high")
        for name in ("arm_tint", "table_tint", "floor_tint", "floor_tile_repeat"):
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
        if (
            not np.isfinite(self.camera_position_jitter)
            or self.camera_position_jitter < 0
        ):
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
    appearance: dict[str, tuple[float, ...]] = field(default_factory=dict)

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
                name: list(position) for name, position in self.clutter_position.items()
            },
            "appearance": {name: list(v) for name, v in self.appearance.items()},
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
        self._base_mat_rgba = model.mat_rgba.copy()
        self._base_mat_texrepeat = model.mat_texrepeat.copy()
        self._base_geom_rgba = model.geom_rgba.copy()
        self._floor_mat = _id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "physai_grid")
        self._table_geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")

    def camera_shift(self, camera_id: int) -> np.ndarray:
        """How far this episode moved a camera from its nominal mount (parent frame)."""
        return self.model.cam_pos[camera_id] - self._base_cam_pos[camera_id]

    def _restore(self) -> None:
        self.model.geom_friction[:] = self._base_geom_friction
        self.model.body_mass[:] = self._base_body_mass
        self.model.light_diffuse[:] = self._base_light_diffuse
        self.model.cam_pos[:] = self._base_cam_pos
        self.model.geom_pos[:] = self._base_geom_pos
        self.model.mat_rgba[:] = self._base_mat_rgba
        self.model.mat_texrepeat[:] = self._base_mat_texrepeat
        self.model.geom_rgba[:] = self._base_geom_rgba

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
            name = (
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id)
                or f"camera_{camera_id}"
            )
            offsets[name] = tuple(float(value) for value in offset)
        clutter_positions: dict[str, tuple[float, float]] = {}
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if not name or not name.startswith("physai_clutter_"):
                continue
            x, y = 0.0, 0.0
            for _ in range(100):
                candidate = np.array(
                    [
                        rng.uniform(*self.config.clutter_x_range),
                        rng.uniform(*self.config.clutter_y_range),
                    ]
                )
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
            appearance=self._randomize_appearance(rng),
        )

    def _randomize_appearance(self, rng: np.random.Generator) -> dict:
        """Tint the robot, table and floor and rescale the floor tiles; what was drawn."""
        config, model, drawn = self.config, self.model, {}

        def tint(name: str) -> np.ndarray | None:
            low, high = getattr(config, name)
            if low == high == 1.0:
                return None  # nothing to draw: keep the rng sequence as it was
            drawn[name] = tuple(float(v) for v in rng.uniform(low, high, size=3))
            return np.asarray(drawn[name])

        if (factor := tint("arm_tint")) is not None:
            for index in range(model.nmat):
                if index != self._floor_mat:
                    model.mat_rgba[index, :3] = np.clip(
                        model.mat_rgba[index, :3] * factor, 0.0, 1.0
                    )
        if (factor := tint("table_tint")) is not None and self._table_geom >= 0:
            model.geom_rgba[self._table_geom, :3] = np.clip(
                model.geom_rgba[self._table_geom, :3] * factor, 0.0, 1.0
            )
        if (factor := tint("floor_tint")) is not None and self._floor_mat >= 0:
            model.mat_rgba[self._floor_mat, :3] = np.clip(
                model.mat_rgba[self._floor_mat, :3] * factor, 0.0, 1.0
            )
        low, high = config.floor_tile_repeat
        if low != high and self._floor_mat >= 0:
            repeat = float(rng.uniform(low, high))
            model.mat_texrepeat[self._floor_mat] = repeat
            drawn["floor_tile_repeat"] = (repeat,)
        return drawn


def _id(model: mujoco.MjModel, kind, name: str) -> int:
    return mujoco.mj_name2id(model, kind, name)


__all__ = [
    "DomainRandomizationConfig",
    "DomainRandomizationEngine",
    "RandomizationMetadata",
]
