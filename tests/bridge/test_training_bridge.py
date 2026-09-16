import json

import numpy as np
import pytest

pytestmark = pytest.mark.integration

from physai.contracts import (
    ALL_JOINT_NAMES,
    ARM_JOINT_NAMES,
    Action,
    Header,
    JointState,
    Observation,
    so101_action_spec,
    so101_action_values,
)
from physai.data import (
    CheckpointMetadata,
    DatasetMetadata,
    EpisodeRecorder,
    EvaluationReport,
    validate_checkpoint_compatibility,
)


def test_so101_action_layout_is_explicit_and_absolute():
    action = Action(
        joint_position=np.arange(5, dtype=np.float64),
        joint_names=ARM_JOINT_NAMES,
    )
    np.testing.assert_array_equal(
        so101_action_values(action, gripper_joint=0.25),
        [0.0, 1.0, 2.0, 3.0, 4.0, 0.25],
    )
    spec = so101_action_spec().to_dict()
    assert spec["metadata"]["joint_names"] == list(ALL_JOINT_NAMES)
    assert spec["metadata"]["absolute"] is True


def test_recorder_writes_versioned_training_metadata(tmp_path):
    observation = Observation(
        joint_state=JointState(
            name=ALL_JOINT_NAMES,
            position=np.zeros(6),
            velocity=np.zeros(6),
            effort=np.zeros(6),
            header=Header(frame_id="base"),
        )
    )
    recorder = EpisodeRecorder(
        tmp_path,
        store_images=False,
        simulator_config={"seed": 4},
        camera_config={"front": {"width": 224, "height": 224}},
        split={"train": [0]},
    )
    recorder.start_episode()
    recorder.record(
        observation,
        Action(joint_position=np.zeros(5), joint_names=ARM_JOINT_NAMES),
        gripper_joint=0.1,
    )
    recorder.end_episode(success=True, extra={"seed": 4})
    meta = json.loads(recorder.write_meta().read_text())
    assert meta["schema_version"] == "physai.dataset.v1"
    assert meta["action_schema"]["names"] == list(ALL_JOINT_NAMES)
    assert meta["seeds"] == [4]
    assert meta["camera_config"]["front"]["width"] == 224


def test_checkpoint_compatibility_checks_nested_contract_fields():
    checkpoint = CheckpointMetadata(
        robot="so101",
        task="pick_place",
        observation_schema={"state": {"shape": [6]}},
        action_schema={
            "schema": "so101.joint_position.v1",
            "names": list(ALL_JOINT_NAMES),
        },
        normalization={"action": {"mean": [0.0]}},
        training_config={"algorithm": "act"},
    )
    validate_checkpoint_compatibility(
        checkpoint,
        {"robot": "so101", "action_schema": {"schema": "so101.joint_position.v1"}},
    )
    with pytest.raises(ValueError, match="robot"):
        validate_checkpoint_compatibility(checkpoint, {"robot": "turtlebot4"})


def test_evaluation_report_separates_failure_modes():
    report = EvaluationReport(
        policy="scripted",
        robot="so101",
        task="pick_place",
        results=(
            {"success": True, "reward": 2.0, "steps": 4, "held_out": True},
            {
                "success": False,
                "reward": -1.0,
                "steps": 8,
                "timeout": True,
                "held_out": True,
            },
            {
                "success": False,
                "reward": -2.0,
                "steps": 2,
                "collision": True,
                "unsafe_action": True,
            },
        ),
    )
    summary = report.summary
    assert summary["success_rate"] == pytest.approx(1 / 3)
    assert summary["collision_count"] == 1
    assert summary["timeout_count"] == 1
    assert summary["unsafe_action_count"] == 1
    assert summary["held_out_success_rate"] == pytest.approx(1 / 2)
