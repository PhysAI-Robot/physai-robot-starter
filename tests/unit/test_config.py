from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_the_legacy_loaders_still_read_the_shipped_files():
    from physai.config import load_sim_config, load_task_config, load_world_config
    from physai.robots.so101 import EnvConfig
    from physai.sim import PickPlaceMinimalSceneConfig

    task = load_task_config(ROOT / "configs" / "tasks" / "so101" / "pick_place.yaml")
    assert task.robot == "so101"
    assert task.task == "pick_place"
    assert task.scene_name == "pick_place_minimal"
    assert isinstance(task.env, EnvConfig)
    assert isinstance(task.env.scene, PickPlaceMinimalSceneConfig)
    assert task.env.scene.cube_names == ("cube",)
    assert task.env.max_steps == 400

    sim = load_sim_config(ROOT / "configs" / "sim_config.yaml")
    assert sim.seed == 0
    assert not sim.domain_randomization.enabled
    assert sim.domain_randomization.friction_scale == (0.9, 1.1)
    assert sim.domain_randomization.mass_scale == (0.95, 1.05)
    assert sim.domain_randomization.lighting_scale == (0.9, 1.1)

    world = load_world_config(ROOT / "configs" / "worlds" / "heterogeneous.yaml")
    assert world.control_hz == 30.0
    assert [item.instance_id for item in world.instances] == ["arm_1", "base_1"]
    assert world.instances[0].robot_name == "so101"
    assert world.instances[1].position == (-0.3, 0.0, 0.1)
