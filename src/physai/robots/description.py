"""Sim-neutral robot description: geometry, frames, cameras, actuators, and
per-simulator overrides owned by the robot, not by any one simulator.

A `RobotDescription` is the seam between a robot's fetched, sim-specific
model files (URDF, MJCF, meshes — never committed, see
`scripts/fetch_assets.py`) and a simulator's own scene builder (today,
`sim.scenes.common.apply_description` for MuJoCo). Every pose here is
expressed in its parent link's frame — position in meters, orientation as a
`(w, x, y, z)` quaternion, angles in radians — so the same numbers can drive
a MuJoCo `MjSpec`, a URDF importer, or a USD stage without change.

Fields a simulator cannot share (MuJoCo's `solref`/`solimp`/`condim`, a
future PhysX material) belong in `sim_overrides`, keyed by simulator name,
and are read only by that simulator's builder — generic code must never
branch on their contents. This module must not import a simulator SDK
(`mujoco`, `isaacsim`, ...); `tests/core/boundaries/test_import_boundaries.py`
enforces that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

Vec3 = tuple[float, float, float]
QuatWXYZ = tuple[float, float, float, float]

IDENTITY_POS: Vec3 = (0.0, 0.0, 0.0)
IDENTITY_QUAT: QuatWXYZ = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Frame:
    """A named, fixed pose attached to a parent link (e.g. an IK target site)."""

    name: str
    parent_link: str
    pos: Vec3 = IDENTITY_POS
    quat: QuatWXYZ = IDENTITY_QUAT


@dataclass(frozen=True)
class CameraDescription:
    """A camera mounted on a parent link.

    `parent_link` may name a link only one fetched model variant provides
    (e.g. a premade camera-mount body); a simulator's builder skips a camera
    whose parent link is absent from the loaded model instead of failing, so
    one description covers every upstream variant.
    """

    name: str
    parent_link: str
    pos: Vec3
    quat: QuatWXYZ
    fovy_deg: float
    width: int
    height: int


@dataclass(frozen=True)
class ContactPad:
    """A collision proxy standing in for a gripper finger's mesh.

    `friction` is MuJoCo's three-component (sliding, torsional, rolling)
    convention; a simulator without that split reads its own equivalent from
    `sim_overrides` or falls back to `friction[0]`.
    """

    name: str
    parent_link: str
    pos: Vec3
    quat: QuatWXYZ
    half_size: Vec3
    friction: tuple[float, float, float]
    disable_parent_mesh_collision: bool = False


@dataclass(frozen=True)
class ActuatorDescription:
    """A joint's drive characteristics, for simulators that do not read them
    from the fetched model file the way MuJoCo reads its own MJCF.
    """

    joint: str
    kind: str = "position"
    stiffness: float = 0.0
    damping: float = 0.0
    force_limit: float = 0.0
    ctrl_range: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class RobotDescription:
    """Sim-neutral robot data, loaded from one `description.yaml` per robot."""

    urdf: str
    mjcf: str
    ee_site: str
    frames: tuple[Frame, ...] = ()
    cameras: tuple[CameraDescription, ...] = ()
    contact_pads: tuple[ContactPad, ...] = ()
    actuators: tuple[ActuatorDescription, ...] = ()
    sim_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Reference values a numeric fit (e.g. `contact_pads`' pose) was derived
    # at — not consumed by any scene builder, kept so the derivation stays
    # reproducible and testable. See `robots/so101/description.yaml`.
    derivation: dict[str, Any] = field(default_factory=dict)

    def mujoco_overrides(self) -> dict[str, Any]:
        return self.sim_overrides.get("mujoco", {})


def _vec3(value: Any) -> Vec3:
    x, y, z = value
    return (float(x), float(y), float(z))


def _quat(value: Any) -> QuatWXYZ:
    w, x, y, z = value
    return (float(w), float(x), float(y), float(z))


def load_robot_description(path: str | Path) -> RobotDescription:
    """Parse a robot's `description.yaml` into a `RobotDescription`.

    `urdf`/`mjcf` are kept as the filenames given in the YAML (relative to
    that robot's fetched asset directory, e.g. `assets/so101/`); resolving
    them against `assets/` is the caller's job (see `robots/so101/scene.py`),
    since this module knows nothing about the repository layout.
    """
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    frames = tuple(
        Frame(
            name=f["name"],
            parent_link=f["parent_link"],
            pos=_vec3(f.get("pos", IDENTITY_POS)),
            quat=_quat(f.get("quat", IDENTITY_QUAT)),
        )
        for f in data.get("frames", [])
    )
    cameras = tuple(
        CameraDescription(
            name=c["name"],
            parent_link=c["parent_link"],
            pos=_vec3(c["pos"]),
            quat=_quat(c["quat"]),
            fovy_deg=float(c["fovy_deg"]),
            width=int(c["width"]),
            height=int(c["height"]),
        )
        for c in data.get("cameras", [])
    )
    contact_pads = tuple(
        ContactPad(
            name=p["name"],
            parent_link=p["parent_link"],
            pos=_vec3(p["pos"]),
            quat=_quat(p["quat"]),
            half_size=_vec3(p["half_size"]),
            friction=_vec3(p["friction"]),
            disable_parent_mesh_collision=bool(
                p.get("disable_parent_mesh_collision", False)
            ),
        )
        for p in data.get("contact_pads", [])
    )
    actuators = tuple(
        ActuatorDescription(
            joint=a["joint"],
            kind=a.get("kind", "position"),
            stiffness=float(a.get("stiffness", 0.0)),
            damping=float(a.get("damping", 0.0)),
            force_limit=float(a.get("force_limit", 0.0)),
            ctrl_range=tuple(float(v) for v in a.get("ctrl_range", (0.0, 0.0))),
        )
        for a in data.get("actuators", [])
    )
    return RobotDescription(
        urdf=data["urdf"],
        mjcf=data["mjcf"],
        ee_site=data["ee_site"],
        frames=frames,
        cameras=cameras,
        contact_pads=contact_pads,
        actuators=actuators,
        sim_overrides=data.get("sim_overrides", {}),
        derivation=data.get("derivation", {}),
    )
