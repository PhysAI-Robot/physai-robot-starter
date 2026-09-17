"""Dataset recording, metadata, evaluation, and Gym integration utilities."""

from __future__ import annotations

from .evaluation import EvaluationReport
from .metadata import (
    CHECKPOINT_SCHEMA_VERSION,
    DATASET_SCHEMA_VERSION,
    CheckpointMetadata,
    DatasetMetadata,
    validate_checkpoint_compatibility,
)
from .recorder import EpisodeRecorder, load_episode

__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "DATASET_SCHEMA_VERSION",
    "CheckpointMetadata",
    "DatasetMetadata",
    "EpisodeRecorder",
    "EvaluationReport",
    "GymnasiumAdapter",
    "load_episode",
    "validate_checkpoint_compatibility",
]


def __getattr__(name: str):
    if name != "GymnasiumAdapter":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        from .gym_env import GymnasiumAdapter
    except ModuleNotFoundError as exc:
        if exc.name != "gymnasium":
            raise
        raise ModuleNotFoundError(
            "GymnasiumAdapter requires the optional training dependency; "
            "install it with `uv sync --extra training`"
        ) from exc
    return GymnasiumAdapter
