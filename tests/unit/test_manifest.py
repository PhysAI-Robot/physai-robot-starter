import re
from pathlib import Path

import pytest
import yaml


def _write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _base_manifest(**overrides) -> dict:
    data = {
        "schema_version": 1,
        "backend": "direct",
        "robots": [{"id": "arm_1", "robot": "so101", "task": "pick_place"}],
    }
    data.update(overrides)
    return data


def test_example_manifests_load_and_validate():
    from physai.config import load_manifest

    manifest = load_manifest("configs/manifests/example_single_so101.yaml")
    assert manifest.robots[0].id == "arm_1"
    assert manifest.task_for(manifest.robots[0]) == "pick_place"
    assert manifest.policy_for(manifest.robots[0]) == "scripted"

    heterogeneous = load_manifest("configs/manifests/example_heterogeneous.yaml")
    assert [r.robot for r in heterogeneous.robots] == ["so101", "turtlebot4"]
    assert heterogeneous.policy_for(heterogeneous.robots[1]) == "idle"


def test_a_single_robot_manifest_is_one_entry_with_a_default_pose(tmp_path):
    from physai.config import load_manifest

    manifest = load_manifest(_write(tmp_path, _base_manifest()))

    assert len(manifest.robots) == 1
    assert manifest.robots[0].pose.position == (0.0, 0.0, 0.0)
    assert manifest.robots[0].pose.quaternion == (1.0, 0.0, 0.0, 0.0)

    data = _base_manifest()
    data["robots"][0]["pose"] = {
        "position": [0.3, 0.0, 0.0],
        "quaternion": [0.0, 0.0, 0.0, 1.0],
    }
    posed = load_manifest(_write(tmp_path, data)).robots[0].pose
    assert posed.position == (0.3, 0.0, 0.0)
    assert posed.quaternion == (0.0, 0.0, 0.0, 1.0)


def test_global_task_and_policy_apply_when_instance_omits_them(tmp_path):
    from physai.config import load_manifest

    data = _base_manifest(
        robots=[{"id": "a", "robot": "so101"}],
        task="pick_place",
        policy="scripted",
    )
    manifest = load_manifest(_write(tmp_path, data))
    assert manifest.task_for(manifest.robots[0]) == "pick_place"
    assert manifest.policy_for(manifest.robots[0]) == "scripted"


def test_robot_config_and_scene_overrides_become_typed_values(tmp_path):
    from physai.config import load_manifest

    data = _base_manifest(
        scene={
            "name": "pick_place_minimal",
            "overrides": {
                "robot_xml": "assets/so101/so101_new_calib_camera.xml",
                "table_pos": [0.3, 0.0, 0.01],
            },
        },
        success_hold_steps=7,
    )
    data["robots"][0]["config"] = {"max_steps": 50, "cube_x_range": [0.2, 0.24]}

    manifest = load_manifest(_write(tmp_path, data))

    assert manifest.robots[0].config == {"max_steps": 50, "cube_x_range": (0.2, 0.24)}
    assert manifest.scene.overrides["table_pos"] == (0.3, 0.0, 0.01)
    assert manifest.scene.overrides["robot_xml"].is_absolute()
    assert manifest.success_hold_steps == 7
    assert manifest.world is None


def test_a_world_block_or_several_robots_make_a_shared_world(tmp_path):
    from physai.config import SessionWorldConfig, load_manifest

    data = _base_manifest(world={"control_hz": 20, "add_floor": False})
    data["robots"][0]["model"] = "assets/so101/so101_new_calib_camera.xml"
    manifest = load_manifest(_write(tmp_path, data))
    assert manifest.world == SessionWorldConfig(
        timestep=0.002, control_hz=20.0, add_floor=False
    )
    assert manifest.robots[0].model.is_absolute()

    several = _base_manifest()
    several["robots"] = [
        {"id": "a", "robot": "so101", "model": "assets/so101/x.xml"},
        {"id": "b", "robot": "turtlebot4", "model": "assets/turtlebot4/x.xml"},
    ]
    assert load_manifest(_write(tmp_path, several)).world == SessionWorldConfig()


def _world_robot() -> dict:
    return {"id": "a", "robot": "so101", "model": "assets/so101/x.xml"}


# (what is wrong, the manifest, the message the error must carry)
INVALID = [
    ("unknown schema version", _base_manifest(schema_version=99), "schema_version"),
    ("empty robots", _base_manifest(robots=[]), "robots"),
    (
        "duplicate ids",
        _base_manifest(
            robots=[{"id": "a", "robot": "so101"}, {"id": "a", "robot": "turtlebot4"}]
        ),
        "duplicate robot id",
    ),
    (
        "unknown robot",
        _base_manifest(robots=[{"id": "a", "robot": "nope"}]),
        "unknown robot",
    ),
    (
        "unknown task",
        _base_manifest(robots=[{"id": "a", "robot": "so101", "task": "nope"}]),
        "unknown task",
    ),
    (
        "unknown policy",
        _base_manifest(robots=[{"id": "a", "robot": "so101", "policy": "nope"}]),
        "unknown policy",
    ),
    (
        "scene incompatible with the robot kind",
        _base_manifest(
            scene={"name": "pick_place_minimal"},
            robots=[{"id": "base_1", "robot": "turtlebot4"}],
        ),
        "incompatible",
    ),
    ("ros2_real backend", _base_manifest(backend="ros2_real"), "not yet implemented"),
    ("unknown backend", _base_manifest(backend="nope"), "unknown backend"),
    ("unknown viewer mode", _base_manifest(viewer={"mode": "nope"}), "viewer.mode"),
    ("world robot without a model", _base_manifest(world={}), "need a 'model'"),
    (
        "zero hold steps",
        _base_manifest(world={}, robots=[_world_robot()], success_hold_steps=0),
        "success_hold_steps",
    ),
    (
        "boolean hold steps",
        _base_manifest(world={}, robots=[_world_robot()], success_hold_steps=True),
        "success_hold_steps",
    ),
    (
        "negative timestep",
        _base_manifest(world={"timestep": -1}, robots=[_world_robot()]),
        "positive",
    ),
    (
        "non-boolean floor",
        _base_manifest(world={"add_floor": "yes"}, robots=[_world_robot()]),
        "boolean",
    ),
    (
        "world that is not a mapping",
        _base_manifest(world="shared", robots=[_world_robot()]),
        "must be a mapping",
    ),
    (
        "robot config that is not a mapping",
        _base_manifest(robots=[{"id": "a", "robot": "so101", "config": ["max_steps"]}]),
        "must be a mapping",
    ),
]


def test_invalid_manifests_fail_loudly(tmp_path):
    from physai.config import load_manifest

    for label, data, message in INVALID:
        try:
            load_manifest(_write(tmp_path, data))
        except ValueError as error:
            assert re.search(message, str(error)), (label, str(error))
        else:
            pytest.fail(f"{label}: the manifest was accepted")
