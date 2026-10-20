"""Newton physics and rendering against MuJoCo, on the SO-101 scene.

Skipped without `newton` (`uv sync` installs it) and a GPU.
"""

import importlib.util

import numpy as np
import pytest
from tests.conftest import requires_assets

if importlib.util.find_spec("newton") is None:
    # Not importorskip: importing newton loads warp, and Isaac Sim must start before
    # warp does (docs/DECISIONS.md, I), so collection must not import it.
    pytest.skip("uv sync installs newton", allow_module_level=True)

from physai.contracts import Action, GripperCommand  # noqa: E402
from physai.robots.so101.mujoco_env import HOME_QPOS, EnvConfig, SO101Env  # noqa: E402
from physai.robots.so101.newton_env import SO101NewtonEnv  # noqa: E402

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.newton,
    pytest.mark.slow,
    requires_assets,
]


def _run(env_class, steps=40):
    env = env_class(EnvConfig(render=False))
    env.reset(seed=3)
    action = Action(joint_position=HOME_QPOS + [0.1, 0.0, -0.1, 0.0, 0.0])
    for _ in range(steps):
        observation = env.step(action)[0]
    return env, observation.joint_state.position


def test_newton_physics_tracks_mujoco():
    reference, expected = _run(SO101Env)
    env, actual = _run(SO101NewtonEnv)
    assert np.abs(actual[:5] - expected[:5]).max() < 0.02
    assert np.abs(env.cube_pos - reference.cube_pos).max() < 0.005
    assert env.robot_spec.metadata["simulator"] == "newton"


def test_newton_physics_resets_to_the_seeded_layout():
    env, _ = _run(SO101NewtonEnv, steps=5)
    env.reset(seed=3)
    first = env.cube_pos.copy()
    env.step(Action(joint_position=HOME_QPOS, gripper=GripperCommand(position=1.0)))
    env.reset(seed=3)
    assert env.cube_pos == pytest.approx(first)


def test_newton_rejects_domain_randomization():
    from physai.sim.mujoco import DomainRandomizationConfig

    with pytest.raises(ValueError, match="does not support domain_randomization"):
        SO101NewtonEnv(
            EnvConfig(
                render=False,
                domain_randomization=DomainRandomizationConfig(enabled=True),
            )
        )


def test_newton_draws_the_robot():
    from physai.robots.so101.factory import make_so101

    env = make_so101(simulator="mujoco", renderer="newton", cameras=("front",))
    env.reset(seed=0)
    shown = env.render_camera("front").astype(float)
    env.data.qpos[env.arm_qadr] = HOME_QPOS + [0.6, 0.0, 0.0, 0.0, 0.0]
    moved = env.render_camera("front").astype(float)
    assert shown.std() > 10  # a scene, not a blank frame
    assert np.abs(shown - moved).mean() > 1  # and it follows the arm
