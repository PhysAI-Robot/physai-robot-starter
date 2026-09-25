from dataclasses import fields, replace

import pytest
import yaml
from conftest import requires_assets, requires_turtlebot_assets

from physai.config import SimulationConfig
from physai.config.compat import (
    manifest_for_robot,
    manifest_from_task_file,
    manifest_from_world_file,
    with_overrides,
)

SIMULATION = SimulationConfig(seed=3)


def test_a_task_file_becomes_a_one_robot_manifest():
    manifest = manifest_from_task_file(
        "configs/tasks/so101/pick_place.yaml", simulation=SIMULATION
    )

    (robot,) = manifest.robots
    assert (robot.id, robot.robot) == ("so101", "so101")
    assert manifest.task == "pick_place"
    assert manifest.scene.name == "pick_place_minimal"
    assert manifest.success_hold_steps == 10
    assert manifest.task_kwargs == {"success_xy_tol": 0.04}
    assert robot.config["cameras"] == ("front", "wrist")
    assert robot.config["max_steps"] == 400
    assert manifest.scene.overrides["cube_names"] == ("cube",)
    assert manifest.simulation.seed == 3
    # settings the manifest owns elsewhere are not left in the robot config
    assert not {"task", "success_xy_tol", "success_hold_steps"} & set(robot.config)


def test_a_task_files_own_seed_wins_over_the_shared_one(tmp_path):
    data = yaml.safe_load(open("configs/tasks/so101/pick_place.yaml"))
    data["env"]["seed"] = 11
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    manifest = manifest_from_task_file(path, simulation=SIMULATION)

    assert manifest.simulation.seed == 11
    assert "seed" not in manifest.robots[0].config


def test_a_task_file_with_mismatched_task_names_fails(tmp_path):
    data = yaml.safe_load(open("configs/tasks/so101/pick_place.yaml"))
    data["env"]["task"] = "sorting"
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValueError, match="task mismatch"):
        manifest_from_task_file(path, simulation=SIMULATION)


def test_a_world_file_becomes_a_shared_world_manifest():
    manifest = manifest_from_world_file(
        "configs/worlds/heterogeneous.yaml", simulation=SIMULATION
    )

    assert manifest.world.control_hz == 30.0
    assert [robot.id for robot in manifest.robots] == ["arm_1", "base_1"]
    assert manifest.robots[0].pose.position == (0.3, 0.0, 0.0)
    assert manifest.robots[0].model.name == "so101_new_calib_camera.xml"


def test_a_bare_manipulator_gets_its_default_task_and_camera():
    manifest = manifest_for_robot("so101", simulation=SIMULATION)

    assert manifest.task == "pick_place"
    assert manifest.scene.overrides == {"camera_width": 640, "camera_height": 480}
    assert manifest.robots[0].config == {"max_steps": 600}


def test_a_bare_base_robot_has_no_task_or_scene():
    manifest = manifest_for_robot("turtlebot4", simulation=SIMULATION)

    assert manifest.task is None
    assert manifest.scene.overrides == {}


def test_overrides_replace_only_what_is_given():
    base = manifest_for_robot("so101", simulation=SIMULATION)

    changed = with_overrides(
        base, seed=9, max_steps=50, camera_size=128, policy="constant"
    )

    assert changed.simulation.seed == 9
    assert changed.robots[0].config["max_steps"] == 50
    assert changed.robots[0].policy == "constant"
    assert changed.scene.overrides == {"camera_width": 128, "camera_height": 128}
    assert with_overrides(base) == base


def test_run_settings_do_not_apply_to_a_shared_world():
    world = manifest_from_world_file(
        "configs/worlds/heterogeneous.yaml", simulation=SIMULATION
    )

    changed = with_overrides(world, max_steps=5, policy="constant")

    assert changed.robots == world.robots


@requires_assets
def test_the_converted_task_file_builds_the_robot_the_legacy_loader_described():
    from physai.config import load_task_config
    from physai.runtime import create_session

    legacy = load_task_config("configs/tasks/so101/pick_place.yaml")
    manifest = manifest_from_task_file(
        "configs/tasks/so101/pick_place.yaml", simulation=SIMULATION
    )

    session = create_session(manifest)
    try:
        built = session.runtime.robot.cfg
        expected = replace(
            legacy.env,
            seed=SIMULATION.seed,
            domain_randomization=SIMULATION.domain_randomization,
        )
        for item in fields(expected):
            if item.name != "scene":
                assert getattr(built, item.name) == getattr(expected, item.name)
        # The env fills in the robot's own scene defaults, so compare only the
        # scene fields the legacy loader actually set.
        for item in fields(expected.scene):
            value = getattr(expected.scene, item.name)
            if isinstance(value, list):  # legacy left some tuple fields as lists
                value = tuple(value)
            if value is not None:
                assert getattr(built.scene, item.name) == value
    finally:
        session.close()
