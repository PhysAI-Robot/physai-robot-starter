"""`--render`: the renderer is chosen apart from the simulator (docs/DECISIONS.md, I)."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from physai.config import load_manifest
from physai.config.compat import with_overrides
from physai.contracts import PolicyTraits
from physai.robots.registry import create_robot
from physai.robots.rendering import attach_renderer
from physai.sim.newton.core import qpos_from_newton, qpos_to_newton

ARM = "configs/manifests/so101_single_cube_fixed_place.yaml"


def test_a_manifest_takes_a_renderer_override_and_defaults_to_none():
    arm = load_manifest(ARM)
    assert arm.renderer is None
    assert with_overrides(arm, simulator="newton", renderer="isaac").renderer == "isaac"
    assert with_overrides(arm, renderer="newton").simulator == "mujoco"


def test_a_renderer_override_is_revalidated():
    base = load_manifest("configs/manifests/turtlebot4.yaml")
    with pytest.raises(ValueError, match="does not support renderer"):
        with_overrides(base, renderer="isaac")  # turtlebot4: mujoco only
    world = load_manifest("configs/manifests/heterogeneous_world.yaml")
    with pytest.raises(ValueError, match="with renderer 'newton' does not support a"):
        with_overrides(world, renderer="newton")


def test_create_robot_rejects_a_renderer_the_robot_lacks():
    with pytest.raises(ValueError, match="does not support renderer 'isaac'"):
        create_robot("turtlebot4", renderer="isaac")


def test_a_newton_env_is_a_mujoco_env_for_policy_support():
    assert PolicyTraits(simulators=("mujoco",)).runs_on("newton")
    assert not PolicyTraits(simulators=("isaac",)).runs_on("newton")
    assert not PolicyTraits().runs_on("isaac")


def test_free_joint_quaternions_swap_between_wxyz_and_xyzw():
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body><freejoint/><geom size='.1'/></body>"
        "<body><joint type='hinge'/><geom size='.1'/></body></worldbody></mujoco>"
    )
    qpos = np.array([1.0, 2.0, 3.0, 0.5, 0.1, 0.2, 0.3, 0.7])  # wxyz after xyz
    q = qpos_to_newton(model, qpos)
    assert q[3:7].tolist() == pytest.approx([0.1, 0.2, 0.3, 0.5])
    assert q[7] == pytest.approx(0.7)
    assert qpos_from_newton(model, q) == pytest.approx(qpos)


def _env(*, scene=True, randomized=False, renderer=None):
    cfg = SimpleNamespace(
        scene=SimpleNamespace(camera_width=8, camera_height=8, video_view=None)
        if scene
        else None,
        domain_randomization=SimpleNamespace(enabled=randomized),
    )
    model = mujoco.MjModel.from_xml_string("<mujoco/>")
    return SimpleNamespace(
        cfg=cfg, model=model, data=mujoco.MjData(model), renderer=renderer
    )


def test_attach_renderer_swaps_the_renderer_and_closes_the_old_one():
    closed = []
    old = SimpleNamespace(close=lambda: closed.append(True))
    env = _env(renderer=old)
    attach_renderer(env, "newton", "mujoco", lambda scene: None)
    assert env.renderer is not old
    assert closed == [True]


def test_attach_renderer_does_nothing_for_the_simulators_own_renderer():
    env = _env(renderer="kept")
    attach_renderer(env, "mujoco", None, lambda scene: None)
    attach_renderer(env, "mujoco", "mujoco", lambda scene: None)
    assert env.renderer == "kept"


def test_attach_renderer_refuses_what_another_renderer_cannot_draw():
    with pytest.raises(ValueError, match="needs a scene"):
        attach_renderer(_env(scene=False), "mujoco", "newton", None)
    with pytest.raises(ValueError, match="cannot draw the configured randomization"):
        attach_renderer(_env(randomized=True), "mujoco", "newton", None)
    with pytest.raises(ValueError, match="unknown renderer"):
        attach_renderer(_env(), "mujoco", "unreal", None)
