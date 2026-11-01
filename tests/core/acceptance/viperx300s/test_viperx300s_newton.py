"""ViperX 300 S on Newton physics against MuJoCo. Skipped without `newton` and a GPU."""

import importlib.util

import numpy as np
import pytest
from tests.conftest import requires_viperx300s_assets

if importlib.util.find_spec("newton") is None:
    # Not importorskip: importing newton loads warp, and Isaac Sim must start before
    # warp does (docs/DECISIONS.md, I), so collection must not import it.
    pytest.skip("uv sync installs newton", allow_module_level=True)

from physai.contracts import Action  # noqa: E402
from physai.robots.viperx300s.mujoco_env import (  # noqa: E402
    HOME_QPOS,
    EnvConfig,
    ViperX300SEnv,
)
from physai.robots.viperx300s.newton_env import ViperX300SNewtonEnv  # noqa: E402

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.newton,
    pytest.mark.slow,
    requires_viperx300s_assets,
]


def _run(env_class, steps=40):
    env = env_class(EnvConfig(render=False))
    env.reset(seed=3)
    action = Action(joint_position=HOME_QPOS + 0.2)
    for _ in range(steps):
        observation = env.step(action)[0]
    return env, observation.joint_state.position


def test_newton_physics_tracks_mujoco():
    reference, expected = _run(ViperX300SEnv)
    env, actual = _run(ViperX300SNewtonEnv)
    assert np.abs(actual - expected).max() < 0.02
    assert np.abs(env.cube_pos - reference.cube_pos).max() < 0.005
    assert env.robot_spec.metadata["simulator"] == "newton"
