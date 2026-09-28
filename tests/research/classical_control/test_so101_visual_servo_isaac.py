"""Parity ladder tier 4 (closed-loop) against real Isaac Sim: `visual_servo`
running its full detect -> approach -> grasp -> transfer -> place loop
(`ROADMAP.md`'s 2E). Skipped wherever `isaacsim` is not installed — `uv
sync --extra isaac` installs it into this project's own venv (see
`docs/adr/0016-isaacsim-as-a-project-extra.md`).

Does not attempt cross-simulator seed/layout parity (`ROADMAP.md`'s 2E
still lists that as open): the cube and target are fixed, configured
positions, not drawn from `robots/so101/layout.py`'s RNG-draw-order
contract, which has no Isaac equivalent yet. This proves the control loop
itself runs end to end against Isaac Sim -- the same policy class, unmodified
control logic -- not a matched success-rate comparison across simulators.

One `SimulationApp` per process is a hard Isaac Sim constraint, so this
module builds its own `SO101IsaacEnv` (with a cube, a front camera, and a
target) via a module-scoped fixture, the same pattern as `test_isaac_env.py`
and `test_so101_grasp_hold_isaac.py`.
"""

from pathlib import Path

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


@pytest.fixture(scope="module")
def env():
    from physai.robots.so101.isaac_env import (
        FrontCameraConfig,
        GraspCubeConfig,
        IsaacEnvConfig,
        SO101IsaacEnv,
    )

    cfg = IsaacEnvConfig(
        usd_out_dir=REPO_ROOT / ".isaac_cache" / "so101_visual_servo_test",
        headless=True,
        render=True,
        cameras=("front", "wrist"),
        max_steps=2000,
        cube=GraspCubeConfig(),
        front_camera=FrontCameraConfig(),
        target_pos=(0.20, -0.10, 0.021),  # matches WorldSceneConfig's own default
    )
    instance = SO101IsaacEnv(cfg)
    try:
        yield instance
    finally:
        instance.close()


def test_visual_servo_detects_the_cube_from_the_front_camera(env):
    """Tier 4, step 1: before running the full control loop, confirm the
    front camera actually sees the (red) cube and `ColorBlobDetector` finds
    it -- a prerequisite the rest of the loop depends on entirely.
    """
    from research.classical_control.so101_visual_servo import ColorBlobDetector

    obs = env.reset(seed=0)
    frame = obs.images["front"]
    assert frame.intrinsics is not None
    assert frame.extrinsics is not None

    detector = ColorBlobDetector()
    feature = detector.detect(frame)
    assert feature is not None, "front camera did not see the cube at all"


def test_visual_servo_runs_the_full_pick_and_place_loop(env):
    """Tier 4: run `SO101VisualServoPolicy` unmodified against Isaac Sim for
    a full episode. Not a success-rate comparison (see this module's
    docstring) -- checks the control loop completes (reaches DONE or exits
    cleanly) and progresses through the whole choreographed phase sequence,
    proving the vision -> IK -> joint-position control path works end to end
    against real Isaac Sim.

    Does not assert the cube is actually picked up and carried to the
    target: real Isaac Sim runs during this test's development showed the
    "gripper" joint's PD drive spins and reports position correctly (via
    `Articulation.get_dof_positions`/`get_dof_velocities`), but the moving
    jaw's own rigid-body prim does not move at all -- confirmed by holding a
    full squeeze for 60 steps at the cube's exact (now pixel-accurate)
    location and lifting: the cube never leaves the ground. That is a
    PhysX/USD joint-articulation issue in the URDF import, tracked as a
    known gap (`ROADMAP.md`'s 2E), not something this policy or env can work
    around; the vision/targeting pipeline itself (`target_xy` vs. the real
    cube position) was independently verified accurate to ~1-2mm.
    """
    from physai.contracts import Action, GripperCommand
    from research.classical_control.so101_visual_servo import (
        SO101VisualServoPolicy,
        VisualServoPhase,
    )

    obs = env.reset(seed=0)
    # `target_plane_z`'s 0.035 default assumes MuJoCo's table_top + cube_half
    # scene (`so101_visual_servo_acceptance.py` relies on that default
    # matching unchanged); `GraspCubeConfig`'s cube sits directly on this
    # env's ground plane instead (no table, see its own docstring), so the
    # default would aim the descend/close waypoint ~2cm above the real cube
    # and mis-triangulate the camera ray-plane intersection by a similar
    # amount -- override it to the cube's actual rest height.
    policy = SO101VisualServoPolicy(env, target_plane_z=env.rest_z)
    policy.reset(obs)

    phases_seen = {policy._phase}
    for _ in range(env.cfg.max_steps):
        action = policy.act(obs)
        obs, *_rest = env.step(action)
        phases_seen.add(policy._phase)
        if policy.done:
            break
    else:
        pytest.fail(
            f"policy did not finish within {env.cfg.max_steps} steps "
            f"(stuck in {policy._phase.name}, failure_reason="
            f"{policy.metrics.failure_reason!r})"
        )

    # The control loop actually progressed through the pick-and-place
    # sequence, not stalled at APPROACH forever failing to detect anything.
    assert VisualServoPhase.CLOSE in phases_seen
    assert VisualServoPhase.LIFT in phases_seen
    assert VisualServoPhase.TRANSFER in phases_seen

    # Release the gripper so a final held-cube slip doesn't leave a false
    # impression; not part of the pass/fail criteria above.
    env.step(
        Action(joint_position=obs.joint_state.position[:5], gripper=GripperCommand(1.0))
    )
