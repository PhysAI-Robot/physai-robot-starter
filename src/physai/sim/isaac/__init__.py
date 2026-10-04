"""Isaac Sim simulation core and `RobotDescription` builder.

Mirrors `physai.sim` for MuJoCo: `core` owns the `SimulationApp`/stepping
lifecycle, `description` applies a robot's sim-neutral `RobotDescription`
to a USD stage, and `scene` adds the generic world primitives (lighting, a
ground plane) a bare URDF import has none of. `robots.so101.isaac_env` is
the only other module allowed to import `isaacsim`/`omni`/`pxr` (see
`pyproject.toml`'s import-linter contract); everything else reaches this
package only through the `RobotPort` it builds.
"""

from __future__ import annotations

from .core import IsaacSimulationCore, close_simulation_app, ensure_simulation_app
from .description import (
    apply_actuators,
    apply_cameras,
    apply_contact_friction,
    apply_frames,
    get_stage,
    import_robot,
    urdf_link_name,
)
from .scene import add_ground_plane, add_studio_lighting

__all__ = [
    "IsaacSimulationCore",
    "add_ground_plane",
    "add_studio_lighting",
    "apply_actuators",
    "apply_cameras",
    "apply_contact_friction",
    "apply_frames",
    "close_simulation_app",
    "ensure_simulation_app",
    "get_stage",
    "import_robot",
    "urdf_link_name",
]
