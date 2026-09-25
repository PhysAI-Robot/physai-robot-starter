"""Pin where reset() places objects for a seed.

The expected values were recorded from the environment before object layout
moved out of `SO101Env.reset`, so this guards that refactor and any later
change to the layout or its random-number order (a seed that suddenly gives a
different scene silently invalidates every recorded dataset and result).
"""

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import requires_assets

GOLDEN = json.loads(
    (Path(__file__).with_name("golden_layouts.json")).read_text(encoding="utf-8")
)

pytestmark = [pytest.mark.acceptance, requires_assets]


def _build(case: str):
    from physai.robots import create_env_config, create_robot
    from physai.robots.registry import scene_defaults
    from physai.sim import (
        DomainRandomizationConfig,
        PickPlaceMinimalSceneConfig,
        SortingMinimalSceneConfig,
    )

    sorting = case.startswith("sorting")
    scene_type = SortingMinimalSceneConfig if sorting else PickPlaceMinimalSceneConfig
    fields = {}
    if case.endswith("fixed_cube_random_target"):
        fields = {"randomize_cube": False, "randomize_target": True}
    clutter = 0
    if case.endswith("domain_randomized"):
        clutter = 2
        fields = {
            "domain_randomization": DomainRandomizationConfig(
                enabled=True,
                clutter_x_range=(0.14, 0.28),
                clutter_y_range=(-0.16, 0.16),
            )
        }
    scene = scene_type(**scene_defaults("so101"), clutter_count=clutter)
    config = create_env_config("so101", scene=scene, render=False, **fields)
    return create_robot("so101", config=config)


def _snapshot(env, seed: int) -> dict:
    env.reset(seed=seed)
    result = {"target_pos": env.target_pos}
    if env.cube_positions:
        result["target_color"] = env.target_color
        result["cubes"] = dict(env.cube_positions)
    else:
        result["cube"] = env.cube_pos
    metadata = env.randomization_metadata
    if metadata.enabled:
        result["clutter"] = dict(metadata.clutter_position)
    return result


def _assert_matches(actual, expected, path=""):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), path
        for key, value in expected.items():
            _assert_matches(actual[key], value, f"{path}/{key}")
    elif isinstance(expected, str):
        assert actual == expected, path
    else:
        np.testing.assert_allclose(
            np.asarray(actual, dtype=float), expected, atol=2e-6, err_msg=path
        )


@pytest.mark.parametrize("case", sorted(GOLDEN))
def test_reset_places_objects_where_it_used_to(case):
    env = _build(case)
    try:
        for seed, expected in GOLDEN[case].items():
            _assert_matches(_snapshot(env, int(seed)), expected, f"{case}/{seed}")
    finally:
        env.close()
