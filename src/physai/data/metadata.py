"""Versioned metadata contracts for datasets and learned checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping


DATASET_SCHEMA_VERSION = "physai.dataset.v1"
CHECKPOINT_SCHEMA_VERSION = "physai.checkpoint.v1"


def _matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and all(
            key in actual and _matches(actual[key], value)
            for key, value in expected.items()
        )
    return json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)


@dataclass(frozen=True)
class DatasetMetadata:
    robot: str
    task: str
    contract_schema: str
    observation_schema: Mapping[str, Any]
    action_schema: Mapping[str, Any]
    simulator_config: Mapping[str, Any] = field(default_factory=dict)
    camera_config: Mapping[str, Any] = field(default_factory=dict)
    seeds: tuple[int, ...] = ()
    split: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = DATASET_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "robot": self.robot,
            "task": self.task,
            "contract_schema": self.contract_schema,
            "observation_schema": dict(self.observation_schema),
            "action_schema": dict(self.action_schema),
            "simulator_config": dict(self.simulator_config),
            "camera_config": dict(self.camera_config),
            "seeds": list(self.seeds),
            "split": dict(self.split),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "DatasetMetadata":
        return cls(
            robot=str(values["robot"]),
            task=str(values["task"]),
            contract_schema=str(values["contract_schema"]),
            observation_schema=values["observation_schema"],
            action_schema=values["action_schema"],
            simulator_config=values.get("simulator_config", {}),
            camera_config=values.get("camera_config", {}),
            seeds=tuple(int(seed) for seed in values.get("seeds", ())),
            split=values.get("split", {}),
            schema_version=str(values.get("schema_version", DATASET_SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class CheckpointMetadata:
    robot: str
    task: str
    observation_schema: Mapping[str, Any]
    action_schema: Mapping[str, Any]
    normalization: Mapping[str, Any]
    training_config: Mapping[str, Any]
    schema_version: str = CHECKPOINT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "robot": self.robot,
            "task": self.task,
            "observation_schema": dict(self.observation_schema),
            "action_schema": dict(self.action_schema),
            "normalization": dict(self.normalization),
            "training_config": dict(self.training_config),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "CheckpointMetadata":
        return cls(
            robot=str(values["robot"]),
            task=str(values["task"]),
            observation_schema=values["observation_schema"],
            action_schema=values["action_schema"],
            normalization=values.get("normalization", {}),
            training_config=values.get("training_config", {}),
            schema_version=str(values.get("schema_version", CHECKPOINT_SCHEMA_VERSION)),
        )


def validate_checkpoint_compatibility(
    actual: CheckpointMetadata | Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    """Reject a checkpoint whose declared contract differs from the runtime."""
    actual_values = (
        actual.to_dict() if isinstance(actual, CheckpointMetadata) else actual
    )
    for key, expected_value in expected.items():
        if key not in actual_values:
            raise ValueError(f"checkpoint metadata is missing {key!r}")
        if not _matches(actual_values[key], expected_value):
            raise ValueError(
                f"checkpoint metadata mismatch for {key!r}: "
                f"expected {expected_value!r}, got {actual_values[key]!r}"
            )


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "DATASET_SCHEMA_VERSION",
    "CheckpointMetadata",
    "DatasetMetadata",
    "validate_checkpoint_compatibility",
]
