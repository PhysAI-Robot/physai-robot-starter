import pytest

from physai.config import SimulationConfig, load_manifest
from physai.config.compat import manifest_for_robot, with_overrides

SIMULATION = SimulationConfig(seed=3)


def test_a_bare_robot_gets_its_defaults_and_command_line_overrides():
    arm = manifest_for_robot("so101", simulation=SIMULATION)
    assert arm.task == "single_cube_fixed_place"
    assert arm.scene.overrides == {}
    assert arm.robots[0].config == {"max_steps": 600}

    base = manifest_for_robot("turtlebot4", simulation=SIMULATION)
    assert base.task is None
    assert base.scene.overrides == {}

    changed = with_overrides(arm, seed=9, max_steps=50, policy="constant")
    assert changed.simulation.seed == 9
    assert changed.robots[0].config["max_steps"] == 50
    assert changed.robots[0].policy == "constant"
    assert with_overrides(arm) == arm

    assert arm.simulation.camera_resolution == "320x240"
    hd = with_overrides(arm, camera_resolution="1280x720")
    assert hd.simulation.camera_resolution == "1280x720"
    with pytest.raises(ValueError, match="unsupported camera resolution"):
        with_overrides(arm, camera_resolution="64x64")

    assert with_overrides(arm, simulator="isaac").simulator == "isaac"
    with pytest.raises(ValueError, match="does not support simulator"):
        with_overrides(base, simulator="isaac")  # turtlebot4: mujoco only


def test_with_overrides_revalidates_a_simulator_override():
    """A bare `dataclasses.replace()` would not re-check `--sim` against a
    shared world; `with_overrides` must, the same way `parse_manifest` does
    at load time (see `physai.config.manifest.validate_manifest`).
    """
    world = load_manifest("configs/manifests/heterogeneous_world.yaml")
    with pytest.raises(ValueError, match="does not support a"):
        with_overrides(world, simulator="isaac")


def test_with_overrides_merges_scene_robot_and_randomization_settings():
    from physai.config import DomainRandomizationConfig

    base = load_manifest("configs/manifests/so101_single_cube_fixed_place.yaml")
    randomization = DomainRandomizationConfig(enabled=True, camera_position_jitter=0.01)
    changed = with_overrides(
        base,
        max_steps=50,
        domain_randomization=randomization,
        scene_overrides={"clutter_count": 2},
        robot_config={"lighting_scale": 0.7},
    )
    assert changed.simulation.domain_randomization is randomization
    assert changed.scene.overrides["clutter_count"] == 2
    assert changed.scene.overrides["cube_half"] == base.scene.overrides["cube_half"]
    config = changed.robots[0].config
    assert (config["max_steps"], config["lighting_scale"]) == (50, 0.7)
    assert config["cameras"] == base.robots[0].config["cameras"]
