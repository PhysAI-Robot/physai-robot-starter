from pathlib import Path


def test_load_minimal_pick_place_config():
    from physai.config import load_task_config
    from physai.robots.so101 import EnvConfig
    from physai.sim import PickPlaceMinimalSceneConfig

    root = Path(__file__).resolve().parents[1]
    config = load_task_config(root / "configs" / "task_pick_place.yaml")

    assert config.robot == "so101"
    assert config.task == "pick_place"
    assert config.scene_name == "pick_place_minimal"
    assert isinstance(config.env, EnvConfig)
    assert isinstance(config.env.scene, PickPlaceMinimalSceneConfig)
    assert config.env.scene.cube_names == ("cube",)
    assert config.env.max_steps == 400