import numpy as np
import pytest
from conftest import requires_assets

pytestmark = [pytest.mark.acceptance, pytest.mark.assets, pytest.mark.slow]


@requires_assets
def test_domain_randomization_is_seeded_bounded_and_restores_baseline():
    from physai.robots.registry import scene_defaults
    from physai.sim import (
        DomainRandomizationConfig,
        DomainRandomizationEngine,
        PickPlaceMinimalSceneConfig,
    )

    model, _ = PickPlaceMinimalSceneConfig(
        clutter_count=2, **scene_defaults("so101")
    ).build_model()
    baseline_friction = model.geom_friction.copy()
    baseline_mass = model.body_mass.copy()
    baseline_lighting = model.light_diffuse.copy()
    disabled = DomainRandomizationEngine(
        model, DomainRandomizationConfig(enabled=False)
    )
    config = DomainRandomizationConfig(
        enabled=True,
        friction_scale=(0.8, 1.2),
        mass_scale=(0.9, 1.1),
        lighting_scale=(0.7, 1.3),
        camera_position_jitter=0.01,
    )
    engine = DomainRandomizationEngine(model, config)
    first = engine.apply(np.random.default_rng(12), seed=12)
    first_friction = model.geom_friction.copy()
    first_mass = model.body_mass.copy()
    first_lighting = model.light_diffuse.copy()
    second = engine.apply(np.random.default_rng(12), seed=12)
    assert first.as_dict() == second.as_dict()
    np.testing.assert_allclose(model.geom_friction, first_friction)
    np.testing.assert_allclose(model.body_mass, first_mass)
    np.testing.assert_allclose(model.light_diffuse, first_lighting)
    assert config.friction_scale[0] <= first.friction_scale <= config.friction_scale[1]
    assert config.mass_scale[0] <= first.mass_scale <= config.mass_scale[1]
    assert config.lighting_scale[0] <= first.lighting_scale <= config.lighting_scale[1]
    for offset in first.camera_position_offset.values():
        assert np.max(np.abs(offset)) <= config.camera_position_jitter
    assert set(first.clutter_position) == {"physai_clutter_0", "physai_clutter_1"}
    for x, y in first.clutter_position.values():
        assert config.clutter_x_range[0] <= x <= config.clutter_x_range[1]
        assert config.clutter_y_range[0] <= y <= config.clutter_y_range[1]
    metadata = disabled.apply(np.random.default_rng(99), seed=99)
    assert not metadata.enabled
    np.testing.assert_allclose(model.geom_friction, baseline_friction)
    np.testing.assert_allclose(model.body_mass, baseline_mass)
    np.testing.assert_allclose(model.light_diffuse, baseline_lighting)


@requires_assets
def test_domain_randomization_metadata_is_recorded_in_episode_info():
    from physai.contracts import Action
    from physai.robots.so101 import EnvConfig, SO101Env
    from physai.sim import DomainRandomizationConfig

    env = SO101Env(
        EnvConfig(
            render=False,
            domain_randomization=DomainRandomizationConfig(
                enabled=True, camera_position_jitter=0.005
            ),
        )
    )
    try:
        observation = env.reset(seed=21)
        _, _, _, _, info = env.step(
            Action(joint_position=observation.joint_state.position[:5])
        )
        assert info["randomization"] == env.randomization_metadata.as_dict()
        assert info["randomization"]["enabled"] is True
        assert info["randomization"]["seed"] == 21
    finally:
        env.close()
