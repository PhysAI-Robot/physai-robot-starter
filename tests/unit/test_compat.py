from dataclasses import fields, replace

import pytest
import yaml
from conftest import requires_assets

from physai.config import SimulationConfig
from physai.config.compat import (
    manifest_for_robot,
    manifest_from_task_file,
    manifest_from_world_file,
    with_overrides,
)

SIMULATION = SimulationConfig(seed=3)
TASK_FILE = "configs/tasks/so101/pick_place.yaml"
WORLD_FILE = "configs/worlds/heterogeneous.yaml"


def _task_file_with(tmp_path, **env):
    with open(TASK_FILE, encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    data["env"].update(env)
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_a_task_file_becomes_a_one_robot_manifest(tmp_path):
    manifest = manifest_from_task_file(TASK_FILE, simulation=SIMULATION)

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

    # a file's own seed wins over the shared one, and moves out of the config
    seeded = manifest_from_task_file(
        _task_file_with(tmp_path, seed=11), simulation=SIMULATION
    )
    assert seeded.simulation.seed == 11
    assert "seed" not in seeded.robots[0].config

    with pytest.raises(ValueError, match="task mismatch"):
        manifest_from_task_file(
            _task_file_with(tmp_path, task="sorting"), simulation=SIMULATION
        )


def test_a_world_file_becomes_a_shared_world_manifest():
    manifest = manifest_from_world_file(WORLD_FILE, simulation=SIMULATION)

    assert manifest.world.control_hz == 30.0
    assert [robot.id for robot in manifest.robots] == ["arm_1", "base_1"]
    assert manifest.robots[0].pose.position == (0.3, 0.0, 0.0)
    assert manifest.robots[0].model.name == "so101_new_calib_camera.xml"
    # run settings mean nothing to a shared world
    assert with_overrides(manifest, max_steps=5, policy="constant").robots == (
        manifest.robots
    )


def test_a_bare_robot_gets_its_defaults_and_command_line_overrides():
    arm = manifest_for_robot("so101", simulation=SIMULATION)
    assert arm.task == "pick_place"
    assert arm.scene.overrides == {"camera_width": 640, "camera_height": 480}
    assert arm.robots[0].config == {"max_steps": 600}

    base = manifest_for_robot("turtlebot4", simulation=SIMULATION)
    assert base.task is None
    assert base.scene.overrides == {}

    changed = with_overrides(
        arm, seed=9, max_steps=50, camera_size=128, policy="constant"
    )
    assert changed.simulation.seed == 9
    assert changed.robots[0].config["max_steps"] == 50
    assert changed.robots[0].policy == "constant"
    assert changed.scene.overrides == {"camera_width": 128, "camera_height": 128}
    assert with_overrides(arm) == arm


@requires_assets
def test_the_converted_task_file_builds_the_robot_the_legacy_loader_described():
    from physai.config import load_task_config
    from physai.runtime import create_session

    legacy = load_task_config(TASK_FILE)
    session = create_session(manifest_from_task_file(TASK_FILE, simulation=SIMULATION))
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


def test_the_shipped_manifests_match_the_legacy_files_they_replace():
    """Delete with the legacy files once their deprecation window ends."""
    from physai.config import load_manifest

    task = manifest_from_task_file(TASK_FILE, simulation=SimulationConfig())
    # The run decides rendering (--video, --serve), so the manifest omits it.
    (robot,) = task.robots
    config = {k: v for k, v in robot.config.items() if k != "render"}
    assert load_manifest("configs/manifests/so101_pick_place.yaml") == replace(
        task, robots=(replace(robot, config=config),)
    )

    world = manifest_from_world_file(WORLD_FILE, simulation=SimulationConfig())
    shipped = load_manifest("configs/manifests/heterogeneous_world.yaml")
    assert shipped == replace(world, viewer=shipped.viewer)
