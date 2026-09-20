from pathlib import Path


def test_load_minimal_pick_place_config():
    from physai.config import load_task_config
    from physai.robots.so101 import EnvConfig
    from physai.sim import PickPlaceMinimalSceneConfig

    root = Path(__file__).resolve().parents[2]
    config = load_task_config(root / "configs" / "tasks" / "so101" / "pick_place.yaml")

    assert config.robot == "so101"
    assert config.task == "pick_place"
    assert config.scene_name == "pick_place_minimal"
    assert isinstance(config.env, EnvConfig)
    assert isinstance(config.env.scene, PickPlaceMinimalSceneConfig)
    assert config.env.scene.cube_names == ("cube",)
    assert config.env.max_steps == 400


def test_load_sim_config_keeps_randomization_disabled():
    from physai.config import load_sim_config

    root = Path(__file__).resolve().parents[2]
    config = load_sim_config(root / "configs" / "sim_config.yaml")

    assert config.seed == 0
    assert not config.domain_randomization.enabled
    assert config.domain_randomization.friction_scale == (0.9, 1.1)
    assert config.domain_randomization.mass_scale == (0.95, 1.05)
    assert config.domain_randomization.lighting_scale == (0.9, 1.1)


def test_load_world_config_builds_heterogeneous_instances():
    from physai.config import load_world_config

    root = Path(__file__).resolve().parents[2]
    config = load_world_config(root / "configs" / "worlds" / "heterogeneous.yaml")

    assert config.control_hz == 30.0
    assert [instance.instance_id for instance in config.instances] == [
        "arm_1",
        "base_1",
    ]
    assert config.instances[0].robot_name == "so101"
    assert config.instances[1].position == (-0.3, 0.0, 0.1)
