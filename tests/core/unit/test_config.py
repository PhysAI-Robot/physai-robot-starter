from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_the_legacy_loaders_still_read_the_shipped_files():
    from physai.config import load_sim_config, load_task_config
    from physai.robots.so101 import EnvConfig
    from physai.sim.mujoco import SingleCubeFixedPlaceSceneConfig

    task = load_task_config(
        ROOT / "configs" / "tasks" / "so101" / "single_cube_fixed_place.yaml"
    )
    assert task.robot == "so101"
    assert task.task == "single_cube_fixed_place"
    assert task.scene_name == "single_cube_fixed_place"
    assert isinstance(task.env, EnvConfig)
    assert isinstance(task.env.scene, SingleCubeFixedPlaceSceneConfig)
    assert task.env.scene.cube_names == ("cube",)
    assert task.env.max_steps == 600

    sim = load_sim_config(ROOT / "configs" / "sim_config.yaml")
    assert sim.seed == 0
    assert not sim.domain_randomization.enabled
    assert sim.domain_randomization.friction_scale == (0.9, 1.1)
    assert sim.domain_randomization.mass_scale == (0.95, 1.05)
    assert sim.domain_randomization.lighting_scale == (0.9, 1.1)
