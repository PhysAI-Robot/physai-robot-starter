"""Typed YAML configuration.

`legacy` holds the pre-manifest per-robot/per-world loaders, re-exported
here unchanged for CLI back-compat. `manifest` is the new unified session
manifest (robots + scene + task + policy + backend + viewer in one file).
"""

from .legacy import (
    DomainRandomizationConfig,
    SimulationConfig,
    TaskConfig,
    WorldConfig,
    load_sim_config,
    load_task_config,
    load_world_config,
)
from .manifest import (
    SessionManifest,
    SessionRobotConfig,
    SessionSceneConfig,
    SessionViewerConfig,
    load_manifest,
)

__all__ = [
    "DomainRandomizationConfig",
    "SessionManifest",
    "SessionRobotConfig",
    "SessionSceneConfig",
    "SessionViewerConfig",
    "SimulationConfig",
    "TaskConfig",
    "WorldConfig",
    "load_manifest",
    "load_sim_config",
    "load_task_config",
    "load_world_config",
]
