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


def test_single_robot_manifest_is_just_one_entry(tmp_path):
    from physai.config import load_manifest

    manifest = load_manifest(_write(tmp_path, _base_manifest()))
    assert len(manifest.robots) == 1
    assert manifest.robots[0].pose.position == (0.0, 0.0, 0.0)
    assert manifest.robots[0].pose.quaternion == (1.0, 0.0, 0.0, 0.0)


def test_pose_position_and_quaternion_round_trip(tmp_path):
    from physai.config import load_manifest

    data = _base_manifest()
    data["robots"][0]["pose"] = {
        "position": [0.3, 0.0, 0.0],
        "quaternion": [0.0, 0.0, 0.0, 1.0],
    }
    manifest = load_manifest(_write(tmp_path, data))
    assert manifest.robots[0].pose.position == (0.3, 0.0, 0.0)
    assert manifest.robots[0].pose.quaternion == (0.0, 0.0, 0.0, 1.0)


def test_unknown_schema_version_fails_loudly(tmp_path):
    from physai.config import load_manifest

    with pytest.raises(ValueError, match="schema_version"):
        load_manifest(_write(tmp_path, _base_manifest(schema_version=99)))


def test_empty_robots_list_is_rejected(tmp_path):
    from physai.config import load_manifest

    with pytest.raises(ValueError, match="robots"):
        load_manifest(_write(tmp_path, _base_manifest(robots=[])))


def test_duplicate_robot_ids_are_rejected(tmp_path):
    from physai.config import load_manifest

    data = _base_manifest(
        robots=[
            {"id": "a", "robot": "so101"},
            {"id": "a", "robot": "turtlebot4"},
        ]
    )
    with pytest.raises(ValueError, match="duplicate robot id"):
        load_manifest(_write(tmp_path, data))


def test_unknown_robot_name_is_rejected(tmp_path):
    from physai.config import load_manifest

    data = _base_manifest(robots=[{"id": "a", "robot": "does-not-exist"}])
    with pytest.raises(ValueError, match="unknown robot"):
        load_manifest(_write(tmp_path, data))


def test_unknown_task_name_is_rejected(tmp_path):
    from physai.config import load_manifest

    data = _base_manifest(
        robots=[{"id": "a", "robot": "so101", "task": "does-not-exist"}]
    )
    with pytest.raises(ValueError, match="unknown task"):
        load_manifest(_write(tmp_path, data))


def test_unknown_policy_name_is_rejected(tmp_path):
    from physai.config import load_manifest

    data = _base_manifest(
        robots=[{"id": "a", "robot": "so101", "policy": "does-not-exist"}]
    )
    with pytest.raises(ValueError, match="unknown policy"):
        load_manifest(_write(tmp_path, data))


def test_scene_incompatible_with_robot_kind_is_rejected(tmp_path):
    from physai.config import load_manifest

    data = _base_manifest(
        scene={"name": "pick_place_minimal"},
        robots=[{"id": "base_1", "robot": "turtlebot4"}],
    )
    with pytest.raises(ValueError, match="incompatible"):
        load_manifest(_write(tmp_path, data))


def test_ros2_real_backend_is_not_yet_implemented(tmp_path):
    from physai.config import load_manifest

    with pytest.raises(ValueError, match="not yet implemented"):
        load_manifest(_write(tmp_path, _base_manifest(backend="ros2_real")))


def test_unknown_backend_is_rejected(tmp_path):
    from physai.config import load_manifest

    with pytest.raises(ValueError, match="unknown backend"):
        load_manifest(_write(tmp_path, _base_manifest(backend="does-not-exist")))


def test_unknown_viewer_mode_is_rejected(tmp_path):
    from physai.config import load_manifest

    data = _base_manifest(viewer={"mode": "does-not-exist"})
    with pytest.raises(ValueError, match="viewer.mode"):
        load_manifest(_write(tmp_path, data))


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
