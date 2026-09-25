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


def test_world_block_makes_a_shared_world_session(tmp_path):
    from physai.config import SessionWorldConfig, load_manifest

    data = _base_manifest(world={"control_hz": 20, "add_floor": False})
    data["robots"][0]["model"] = "assets/so101/so101_new_calib_camera.xml"

    manifest = load_manifest(_write(tmp_path, data))

    assert manifest.world == SessionWorldConfig(
        timestep=0.002, control_hz=20.0, add_floor=False
    )
    assert manifest.robots[0].model.is_absolute()


def test_several_robots_share_one_default_world(tmp_path):
    from physai.config import SessionWorldConfig, load_manifest

    data = _base_manifest()
    data["robots"] = [
        {"id": "a", "robot": "so101", "model": "assets/so101/x.xml"},
        {"id": "b", "robot": "turtlebot4", "model": "assets/turtlebot4/x.xml"},
    ]

    assert load_manifest(_write(tmp_path, data)).world == SessionWorldConfig()


def test_world_robots_must_name_their_model(tmp_path):
    from physai.config import load_manifest

    with pytest.raises(ValueError, match="need a 'model'"):
        load_manifest(_write(tmp_path, _base_manifest(world={})))


@pytest.mark.parametrize(
    "extra",
    [
        {"success_hold_steps": 0},
        {"success_hold_steps": True},
        {"world": {"timestep": -1}},
        {"world": {"add_floor": "yes"}},
        {"world": "shared"},
    ],
)
def test_invalid_world_and_hold_settings_fail_loudly(tmp_path, extra):
    from physai.config import load_manifest

    data = _base_manifest(**extra)
    data["robots"][0]["model"] = "assets/so101/x.xml"

    with pytest.raises(ValueError):
        load_manifest(_write(tmp_path, data))


def test_robot_config_must_be_a_mapping(tmp_path):
    from physai.config import load_manifest

    data = _base_manifest()
    data["robots"][0]["config"] = ["max_steps"]

    with pytest.raises(ValueError, match="must be a mapping"):
        load_manifest(_write(tmp_path, data))
