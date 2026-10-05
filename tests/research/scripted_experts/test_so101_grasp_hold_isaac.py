"""Parity ladder tier 3 (contact) against real Isaac Sim: a cube grasp-hold
test, comparing slip against the MuJoCo no-slip baseline in
`test_so101_grasp_hold.py` (`ROADMAP.md`'s 2E). Skipped wherever `isaacsim`
is not installed — `uv sync --extra isaac` installs it into this project's
own venv (see `docs/adr/0016-isaacsim-as-a-project-extra.md`).

Deliberately not a port of `SO101PickPlaceExpert`: that class is tuned
against MuJoCo-specific behavior (raw `MjData` contact iteration in
`_cube_grasped`, `env.data`-keyed kinematics calls) with no Isaac
equivalent for the contact-iteration part, and its tuning is documented as
sensitive to small changes (see its own module docstring). This is a
smaller, open-loop grasp-lift-hold sequence built directly from
`ArmKinematics.ik_pinch`/`qpos_to_site_pose` (both already joint-position
based, needing no MuJoCo live `data`), reusing only what tier 3 needs.

One `SimulationApp` per process is a hard Isaac Sim constraint, so this
module builds its own `SO101IsaacEnv` (with a cube) via a module-scoped
fixture, the same pattern as `test_isaac_env.py`.
"""

from pathlib import Path

import numpy as np
import pytest

from tests.conftest import requires_assets

_isaacsim = pytest.importorskip(
    "isaacsim", reason="uv sync --extra isaac installs isaacsim"
)
if not hasattr(_isaacsim, "SimulationApp"):
    # See test_isaac_env.py's matching guard: a stale, empty `isaacsim`
    # namespace package can outlive `uv sync` dropping the `isaac` extra.
    pytest.skip(
        "isaacsim package is present but empty (stale extra switch?)",
        allow_module_level=True,
    )

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.isaac,
    pytest.mark.slow,
    requires_assets,
]

REPO_ROOT = Path(__file__).resolve().parents[3]

# Mirrors so101_pick_place_expert.ExpertConfig's own values -- same
# normalized gripper convention (0 closed, 1 open), same reasoning (a
# shallow first-touch settles with too little grip force; a deeper squeeze
# is needed to actually hold the cube against its own weight).
_GRIP_OPEN = 0.55
_GRIP_SQUEEZE = 0.06
_HOVER_HEIGHT = 0.045
_MAX_JOINT_STEP = 1.2 / 30.0  # rad per control tick, visual_servo's rate
_LIFT_HEIGHT = 0.06
_STEPS_PER_PHASE = (
    60  # 2 s at 30 Hz -- generous given the 0.3 rad/3 s settling tier-2 verified
)


@pytest.fixture(scope="module")
def env():
    from physai.robots.so101.isaac_env import IsaacEnvConfig, SO101IsaacEnv
    from physai.sim.mujoco import SingleCubeFixedPlaceSceneConfig

    cfg = IsaacEnvConfig(
        usd_out_dir=REPO_ROOT / ".isaac_cache" / "so101_grasp_hold_test",
        headless=True,
        render=False,
        cameras=(),
        max_steps=1000,
        # The same table scene the MuJoCo grasp-hold test grasps in.
        scene=SingleCubeFixedPlaceSceneConfig(),
        randomize_cube=False,
    )
    instance = SO101IsaacEnv(cfg)
    try:
        yield instance
    finally:
        instance.close()


def _move_toward(env, target_xyz, grip: float, steps: int) -> np.ndarray:
    """Command the arm toward an IK solution for `steps` control ticks."""
    from physai.contracts import Action, GripperCommand
    from physai.robots.so101.kinematics import TOP_DOWN

    q = env._arm_qpos()
    q_cmd = q.copy()
    for _ in range(steps):
        result = env.kin.ik_pinch(target_xyz, TOP_DOWN, q_init=q)
        q = result.qpos
        # Rate-limit the command like visual_servo's JointRateLimiter: jumping
        # straight to the IK solution swings the arm fast enough to sweep the
        # cube off the table on the way to the hover point.
        q_cmd = q_cmd + np.clip(q - q_cmd, -_MAX_JOINT_STEP, _MAX_JOINT_STEP)
        env.step(Action(joint_position=q_cmd, gripper=GripperCommand(position=grip)))
    return q_cmd


def grasp_and_lift(env) -> np.ndarray:
    """Grasp the cube and lift it clear of the ground; return final joint positions."""
    env.reset(seed=0)
    cube = env.cube_pos
    hover = np.array([cube[0], cube[1], cube[2] + _HOVER_HEIGHT])
    grasp = np.array([cube[0], cube[1], cube[2]])
    lift = np.array([cube[0], cube[1], cube[2] + _LIFT_HEIGHT])

    _move_toward(env, hover, _GRIP_OPEN, _STEPS_PER_PHASE)
    _move_toward(env, grasp, _GRIP_OPEN, _STEPS_PER_PHASE)
    _move_toward(env, grasp, _GRIP_SQUEEZE, _STEPS_PER_PHASE)
    return _move_toward(env, lift, _GRIP_SQUEEZE, _STEPS_PER_PHASE)


def cube_in_gripper(env) -> float:
    """Cube centre along the gripper's approach axis, in millimetres.

    Mirrors `test_so101_grasp_hold.py`'s `cube_in_gripper` exactly (offset
    from the end-effector site, projected onto the site's own rotation),
    substituting `qpos_to_site_pose`'s FK-from-observed-joint-positions for
    MuJoCo's live `site_xpos`/`site_xmat` reads.
    """
    site_pos, site_rotation = env.kin.qpos_to_site_pose(env._arm_qpos())
    offset = env.cube_pos - site_pos
    return 1000.0 * float((offset @ site_rotation)[0])


@requires_assets
def test_a_held_cube_does_not_creep_out_of_the_fingers(env):
    """Isaac-side counterpart to `test_so101_grasp_hold.py`'s test of the
    same name. No PhysX contact-report API is wired up yet (see the plan
    this test accompanies), so "grasped" is confirmed the same way
    `so101_pick_place_expert._cube_lifted()` already does on the MuJoCo
    side: by height, not contact truth.
    """
    from physai.contracts import Action, GripperCommand

    spawn_height = env.rest_z
    grasp_and_lift(env)
    assert env.cube_pos[2] > spawn_height + 0.02  # lifted, not left on the ground

    hold = env._arm_qpos()
    start = cube_in_gripper(env)
    for _ in range(int(12 * env.cfg.control_hz)):
        env.step(
            Action(joint_position=hold, gripper=GripperCommand(position=_GRIP_SQUEEZE))
        )

    assert abs(cube_in_gripper(env) - start) < 1.0  # millimetres, over 12 s
    assert env.cube_pos[2] > spawn_height + 0.02  # still lifted
