"""The manipulation workspace both simulators build: table, target, cubes, camera.

Neutral module, like `physai.sim.studio`: `physai.sim.mujoco` and
`physai.sim.isaac` stay independent backends (an import-linter contract), so
the description they both read lives here and each backend owns only how it
builds it (`sim.mujoco.scenes.common.build_manipulation_spec`,
`sim.isaac.scene.add_workspace`). Mirrors `robots.description` for the robot.
The physical constants below are the values both builders use, so a
difference between simulators is a physics gap, not a drifted default.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..contracts import DEFAULT_CAMERA_RESOLUTION, parse_camera_resolution

# MuJoCo's (sliding, torsional, rolling) friction; PhysX has one coefficient,
# so the Isaac builder uses the first component.
TABLE_FRICTION: tuple[float, float, float] = (1.0, 0.005, 0.0001)
CUBE_FRICTION: tuple[float, float, float] = (1.2, 0.01, 0.0005)
# The target pad is a visual-only disc on the table.
TARGET_RGBA: tuple[float, float, float, float] = (0.2, 0.7, 0.35, 0.55)
FRONT_CAMERA_FOVY_DEG: float = 48.0


@dataclass(frozen=True)
class CubeSpec:
    """One dynamic, graspable cube."""

    name: str
    position: tuple[float, float, float]
    half_size: float
    mass: float
    rgba: tuple[float, float, float, float]


@dataclass
class WorkspaceConfig:
    """Robot-independent, simulator-independent workspace settings."""

    table_size: tuple[float, float, float] = (0.20, 0.25, 0.01)
    table_pos: tuple[float, float, float] = (0.30, 0.0, 0.01)
    target_pos: tuple[float, float, float] = (0.20, -0.10, 0.021)
    target_radius: float = 0.035
    # One of `physai.contracts.CAMERA_RESOLUTIONS`; the pixel size below is
    # derived from it (so it stays in dataset metadata) and not an argument.
    camera_resolution: str = DEFAULT_CAMERA_RESOLUTION
    camera_width: int = field(default=0, init=False)
    camera_height: int = field(default=0, init=False)
    front_cam_pos: tuple[float, float, float] = (0.62, 0.0, 0.38)
    front_cam_xyaxes: tuple[float, ...] = (0.0, 1.0, 0.0, -0.45, 0.0, 0.9)

    def __post_init__(self) -> None:
        self.camera_width, self.camera_height = parse_camera_resolution(
            self.camera_resolution
        )

    def cubes(self) -> tuple[CubeSpec, ...]:
        """The cubes this workspace places; a task scene overrides this."""
        return ()


__all__ = [
    "CUBE_FRICTION",
    "FRONT_CAMERA_FOVY_DEG",
    "TABLE_FRICTION",
    "TARGET_RGBA",
    "CubeSpec",
    "WorkspaceConfig",
]
