"""The cube-start file of O2 (study 01): format, and that it changes nothing when off."""

import json

import numpy as np
import pytest


def _write(tmp_path, body):
    path = tmp_path / "start.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def test_offset_is_added_to_the_drawn_xy(tmp_path):
    from physai.robots.cube_start import CubeStart

    start = CubeStart.load(_write(tmp_path, {"offset_xy": [0.001, 0.0086]}))
    assert start.apply(1000, (0.2, 0.05)) == pytest.approx((0.201, 0.0586))
    assert start.apply(None, (0.2, 0.05)) == pytest.approx((0.201, 0.0586))


def test_table_replaces_the_drawn_xy_and_needs_a_known_seed(tmp_path):
    from physai.robots.cube_start import CubeStart

    start = CubeStart.load(
        _write(tmp_path, {"source": "x", "xy_by_seed": {"1000": [0.2, 0.05]}})
    )
    assert start.apply(1000, (0.9, 0.9)) == (0.2, 0.05)
    with pytest.raises(KeyError):
        start.apply(1001, (0.2, 0.05))
    with pytest.raises(ValueError):
        start.apply(None, (0.2, 0.05))


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"offset_xy": [0, 0], "xy_by_seed": {"1": [0, 0]}},
        {"offset_xy": [0, 0, 0]},
        {"offset_xy": [0, 0], "typo": 1},
    ],
)
def test_a_malformed_file_is_refused(tmp_path, body):
    from physai.robots.cube_start import CubeStart

    with pytest.raises(ValueError):
        CubeStart.load(_write(tmp_path, body))


def _envs():
    mujoco = pytest.importorskip("mujoco")
    assert mujoco
    from physai.robots.so101 import EnvConfig as SO101Config
    from physai.robots.so101 import SO101Env
    from physai.robots.viperx300s import EnvConfig as ViperXConfig
    from physai.robots.viperx300s import ViperX300SEnv

    return [(SO101Env, SO101Config), (ViperX300SEnv, ViperXConfig)]


@pytest.mark.parametrize("index", [0, 1])
def test_off_or_zero_offset_changes_nothing_and_a_table_moves_only_the_cube(
    tmp_path, index
):
    env_cls, config_cls = _envs()[index]
    plain = env_cls(config_cls(render=False, randomize_target=True))
    plain.reset(seed=1003)
    cube0, target0 = plain.cube_pos.copy(), plain.target_pos.copy()
    qpos0 = plain.data.qpos.copy()
    plain.close()

    zero = _write(tmp_path, {"offset_xy": [0.0, 0.0]})
    same = env_cls(
        config_cls(render=False, randomize_target=True, cube_start=str(zero))
    )
    same.reset(seed=1003)
    np.testing.assert_array_equal(same.data.qpos, qpos0)
    same.close()

    table = _write(
        tmp_path, {"xy_by_seed": {"1003": [cube0[0] + 0.003, cube0[1] + 0.0086]}}
    )
    moved = env_cls(
        config_cls(render=False, randomize_target=True, cube_start=str(table))
    )
    moved.reset(seed=1003)
    np.testing.assert_allclose(
        moved.cube_pos[:2], cube0[:2] + [0.003, 0.0086], atol=1e-9
    )
    np.testing.assert_allclose(moved.cube_pos[2], cube0[2], atol=1e-9)
    np.testing.assert_array_equal(moved.target_pos, target0)
    moved.close()
