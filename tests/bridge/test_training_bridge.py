import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.integration

from physai.contracts import (
    Action,
    Header,
    JointState,
    Observation,
)
from physai.data import (
    CheckpointMetadata,
    EpisodeRecorder,
    EvaluationReport,
    load_episode,
    validate_checkpoint_compatibility,
)
from physai.robots.so101.contracts import (
    ALL_JOINT_NAMES,
    ARM_JOINT_NAMES,
    so101_action_schema,
    so101_action_spec,
    so101_action_values,
    so101_observation_schema,
    so101_observation_spec,
)
from physai.robots.turtlebot.contracts import turtlebot4_training_contract


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


def test_robot_training_contracts_define_distinct_action_and_camera_layouts():
    so101 = so101_action_spec()
    turtlebot = turtlebot4_training_contract()

    assert so101.metadata["schema"] == "so101.joint_position.v1"
    assert turtlebot.action_spec.metadata["schema"] == "turtlebot4.twist.v1"
    assert tuple(camera.name for camera in turtlebot.observation_spec.cameras) == (
        "free",
    )
    assert turtlebot.action_decoder is not None


def test_so101_observation_schema_is_canonical_and_dataset_shaped():
    spec = so101_observation_spec(
        camera_config={"front": {"width": 320, "height": 240}}
    )
    assert spec.metadata["schema"] == "so101.observation.v1"
    assert spec.cameras[0].shape == (240, 320, 3)
    schema = so101_observation_schema(
        camera_config={"front": {"width": 320, "height": 240}}
    )
    assert schema["observation.state"]["shape"] == [6]
    assert schema["observation.images.front"]["shape"] == [240, 320, 3]


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
        task_name="pick_place",
        simulator_config={"seed": 4},
        camera_config={"front": {"width": 224, "height": 224}},
        split={"train": [0]},
        action_encoder=lambda action, gripper_joint: (
            so101_action_values(action, gripper_joint=gripper_joint or 0.0),
            ALL_JOINT_NAMES,
        ),
        action_schema=so101_action_schema(),
        observation_schema=so101_observation_schema(
            camera_config={"front": {"width": 224, "height": 224}}
        ),
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
    assert meta["task_name"] == "pick_place"
    assert meta["camera_config"]["front"]["width"] == 224


def _joint_observation() -> Observation:
    return Observation(
        joint_state=JointState(
            name=("joint",),
            position=np.zeros(1),
            velocity=np.zeros(1),
            effort=np.zeros(1),
            header=Header(frame_id="base"),
        )
    )


def test_recorder_stores_environment_state_when_configured(tmp_path):
    recorder = EpisodeRecorder(tmp_path, store_images=False, environment_state_dim=3)
    recorder.start_episode()
    for step in range(2):
        recorder.record(
            _joint_observation(),
            Action(joint_position=np.zeros(1)),
            environment_state=np.array([step, 0.5, -1.0]),
        )
    path = recorder.end_episode(success=True)

    episode = load_episode(path)
    np.testing.assert_array_equal(
        episode["observation.environment_state"], [[0, 0.5, -1.0], [1, 0.5, -1.0]]
    )
    meta = json.loads(recorder.write_meta().read_text())
    assert meta["features"]["observation.environment_state"]["shape"] == [3]


def test_recorder_without_environment_state_keeps_the_original_layout(tmp_path):
    recorder = EpisodeRecorder(tmp_path, store_images=False)
    recorder.start_episode()
    recorder.record(_joint_observation(), Action(joint_position=np.zeros(1)))
    episode = load_episode(recorder.end_episode(success=True))

    assert set(episode) == {"observation.state", "action", "reward", "done", "phase"}
    meta = json.loads(recorder.write_meta().read_text())
    assert "observation.environment_state" not in meta["features"]


def test_recorder_rejects_mismatched_environment_state(tmp_path):
    configured = EpisodeRecorder(
        tmp_path / "a", store_images=False, environment_state_dim=3
    )
    configured.start_episode()
    action = Action(joint_position=np.zeros(1))
    with pytest.raises(ValueError, match="3 values"):
        configured.record(_joint_observation(), action)
    with pytest.raises(ValueError, match="3 values"):
        configured.record(_joint_observation(), action, environment_state=np.zeros(2))

    plain = EpisodeRecorder(tmp_path / "b", store_images=False)
    plain.start_episode()
    with pytest.raises(ValueError, match="environment_state_dim"):
        plain.record(_joint_observation(), action, environment_state=np.zeros(3))


def test_load_episode_can_read_only_the_requested_keys(tmp_path):
    recorder = EpisodeRecorder(tmp_path, store_images=False, environment_state_dim=2)
    recorder.start_episode()
    recorder.record(
        _joint_observation(),
        Action(joint_position=np.zeros(1)),
        environment_state=np.array([1.0, 2.0]),
    )
    path = recorder.end_episode(success=True)

    only_state = load_episode(path, keys=("observation.environment_state",))

    assert list(only_state) == ["observation.environment_state"]
    np.testing.assert_array_equal(
        only_state["observation.environment_state"], [[1.0, 2.0]]
    )


def test_recorder_discard_episode_writes_nothing(tmp_path):
    recorder = EpisodeRecorder(tmp_path, store_images=False)
    recorder.start_episode()
    recorder.record(_joint_observation(), Action(joint_position=np.zeros(1)))
    recorder.discard_episode()

    assert recorder.end_episode(success=True) is None
    assert list(tmp_path.glob("*.npz")) == []


def test_recorder_writes_path_valued_scene_config_fields(tmp_path):
    # SceneConfig.robot_xml is a Path; meta.json used to crash on it.
    recorder = EpisodeRecorder(
        tmp_path,
        store_images=False,
        scene_config={"robot_xml": Path("assets") / "so101" / "robot.xml"},
    )

    meta = json.loads(recorder.write_meta().read_text())

    assert meta["scene_config"]["robot_xml"] == "assets/so101/robot.xml"


def test_recorder_records_scene_identity(tmp_path):
    recorder = EpisodeRecorder(
        tmp_path,
        store_images=False,
        scene_name="pick_place_minimal",
        scene_config={"cube_pos": [0.2, 0.0, 0.034]},
    )
    meta = json.loads(recorder.write_meta().read_text())
    assert meta["scene_name"] == "pick_place_minimal"
    assert meta["scene_config"]["cube_pos"] == [0.2, 0.0, 0.034]


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


def test_checkpoint_compatibility_rejects_unknown_schema_version():
    checkpoint = CheckpointMetadata(
        robot="so101",
        task="pick_place",
        observation_schema={"observation.state": {"shape": [6]}},
        action_schema={"schema": "so101.joint_position.v1"},
        normalization={},
        training_config={},
        schema_version="physai.checkpoint.v0",
    )
    with pytest.raises(ValueError, match="unsupported checkpoint metadata schema"):
        validate_checkpoint_compatibility(checkpoint, {})


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
