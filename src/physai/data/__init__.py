"""Dataset recording, metadata, evaluation, and Gym integration utilities."""

from .gym_env import GymnasiumAdapter
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
    "EvaluationReport",
    "EpisodeRecorder",
    "GymnasiumAdapter",
    "load_episode",
    "validate_checkpoint_compatibility",
]
