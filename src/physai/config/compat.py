"""Build a session manifest from a bare robot name, and apply command-line overrides.

A bare `--robot` is the one input `scripts/run_sim.py` still turns into a manifest
(`manifest_for_robot`); `with_overrides` applies the flags both run scripts share.
Both go through `parse_manifest`/`validate_manifest`, so the rest of the code only
ever sees a validated `SessionManifest`.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from ..contracts import parse_camera_resolution
from ..robots.registry import default_task
from .legacy import DomainRandomizationConfig, SimulationConfig
from .manifest import SCHEMA_VERSION, SessionManifest, parse_manifest, validate_manifest

# Episode length of a bare `--robot` run that gives no --max-steps.
BARE_ROBOT_MAX_STEPS = 600


def manifest_for_robot(robot: str, *, simulation: SimulationConfig) -> SessionManifest:
    """The manifest of `run_sim.py --robot <name>` without a config file."""
    task = default_task(robot)
    manifest_data: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "robots": [
            {
                "id": robot,
                "robot": robot,
                "config": {"max_steps": BARE_ROBOT_MAX_STEPS},
            }
        ],
    }
    if task is not None:
        manifest_data["task"] = task
    return _with_simulation(
        parse_manifest(manifest_data, Path(f"<--robot {robot}>")), simulation, None
    )


def with_overrides(
    manifest: SessionManifest,
    *,
    seed: int | None = None,
    max_steps: int | None = None,
    camera_resolution: str | None = None,
    policy: str | None = None,
    simulator: str | None = None,
    domain_randomization: DomainRandomizationConfig | None = None,
    scene_overrides: dict[str, Any] | None = None,
    robot_config: dict[str, Any] | None = None,
) -> SessionManifest:
    """The manifest with command-line overrides applied; ``None`` keeps a value.

    ``scene_overrides`` and ``robot_config`` are merged over the manifest's own
    scene overrides and every robot's config.
    """
    changes: dict[str, Any] = {}
    simulation = manifest.simulation
    if domain_randomization is not None:
        simulation = replace(simulation, domain_randomization=domain_randomization)
    if scene_overrides:
        changes["scene"] = replace(
            manifest.scene,
            overrides={**manifest.scene.overrides, **scene_overrides},
        )
    if seed is not None:
        simulation = replace(simulation, seed=seed)
    if camera_resolution is not None:
        parse_camera_resolution(camera_resolution)
        simulation = replace(simulation, camera_resolution=camera_resolution)
    if simulation is not manifest.simulation:
        changes["simulation"] = simulation
    robot_updates = {
        **(robot_config or {}),
        **({} if max_steps is None else {"max_steps": max_steps}),
    }
    if manifest.world is None and (robot_updates or policy is not None):
        changes["robots"] = tuple(
            replace(
                robot,
                config={**robot.config, **robot_updates},
                policy=robot.policy if policy is None else policy,
            )
            for robot in manifest.robots
        )
    if simulator is not None:
        changes["simulator"] = simulator
    updated = replace(manifest, **changes)
    if simulator is not None:
        # A bare `replace()` skips registry/cross-field checks; --sim can
        # select a simulator a robot does not support or that conflicts with
        # world/backend/viewer, so the override is re-validated the same way
        # the manifest itself was at load time.
        validate_manifest(updated, Path("<--sim override>"))
    return updated


def _with_simulation(
    manifest: SessionManifest, simulation: SimulationConfig, seed: int | None
) -> SessionManifest:
    """Attach the shared simulation settings; a file's own seed wins over them."""
    if seed is not None:
        simulation = replace(simulation, seed=seed)
    return replace(manifest, simulation=simulation)


__all__ = [
    "BARE_ROBOT_MAX_STEPS",
    "manifest_for_robot",
    "with_overrides",
]
