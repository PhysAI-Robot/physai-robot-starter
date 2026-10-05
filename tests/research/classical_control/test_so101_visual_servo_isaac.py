"""Parity ladder tier 4 (closed-loop) against real Isaac Sim: `visual_servo`
running its full detect -> approach -> grasp -> transfer -> place loop
(`ROADMAP.md`'s 2E). Skipped wherever `isaacsim` is not installed — `uv
sync --extra isaac` installs it into this project's own venv (see
`docs/adr/0016-isaacsim-as-a-project-extra.md`).

The cube and target here are fixed, configured positions (the low-level
`cube`/`table`/`target_pos` config), which makes this the quick check that the
control loop runs end to end against Isaac Sim and delivers the cube with the
same policy class, control logic, and parameters as MuJoCo, in the same table
scene. The matched, per-seed comparison over the shared scene config is
`scripts/eval_policy.py --sim isaac` with `scripts/compare_evaluations.py`
(results in `research/classical_control/FINDINGS.md`); the per-seed cube
layout itself is pinned by `tests/core/acceptance/so101/test_isaac_layout.py`.

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
    from physai.robots.so101.isaac_env import IsaacEnvConfig, SO101IsaacEnv
    from physai.sim.mujoco import SingleCubeFixedPlaceSceneConfig

    cfg = IsaacEnvConfig(
        usd_out_dir=REPO_ROOT / ".isaac_cache" / "so101_visual_servo_test",
        headless=True,
        render=True,
        cameras=("front", "wrist"),
        max_steps=2000,
        scene=SingleCubeFixedPlaceSceneConfig(),
        randomize_cube=False,
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
    """Tier 4: run `SO101VisualServoPolicy` against Isaac Sim for a full
    episode and assert it *delivers* the cube: detect -> approach -> grasp ->
    lift -> transfer -> place, ending within 4 cm of the target (the same
    bound MuJoCo's `so101_visual_servo_acceptance.py` uses). Not a
    success-rate comparison (see this module's docstring): one fixed layout.

    What made this pass, in order (`research/classical_control/FINDINGS.md`):
    (1) `act()`'s CLOSE/RELEASE settle check compared the gripper to its
    *ramping* command, which Isaac tracks with no lag, so CLOSE exited ~21%
    closed; it now waits for the ramp and for a real settle. (2) Isaac gripped
    with the raw jaw mesh while MuJoCo grips with fitted 12 x 12 x 6 mm pad
    boxes -- `sim.isaac.description.apply_contact_pad_colliders` now builds
    the same pads from the shared `contact_pads` spec. Five parameter
    experiments before (2) (force cap, solver iterations, friction, pad
    softness) all failed or made tier 3 worse; the geometry was the cause.

    The policy is built with its defaults, exactly as in MuJoCo.
    """
    from physai.contracts import Action, GripperCommand
    from research.classical_control.so101_visual_servo import (
        SO101VisualServoPolicy,
        VisualServoPhase,
    )

    obs = env.reset(seed=0)
    policy = SO101VisualServoPolicy(
        env,
    )
    policy.reset(obs)
    spawn_z = env.cube_pos[2]
    max_cube_z = spawn_z

    phases_seen = {policy._phase}
    for _ in range(env.cfg.max_steps):
        action = policy.act(obs)
        obs, *_rest = env.step(action)
        phases_seen.add(policy._phase)
        max_cube_z = max(max_cube_z, float(env.cube_pos[2]))
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

    # The gripper actually lifted the cube (it is carried, not dragged).
    assert max_cube_z > spawn_z + 0.02, (
        f"cube only rose {max_cube_z - spawn_z:.4f}m above its rest height "
        "-- the gripper likely never got a real grip on it"
    )

    # Delivery: within 4 cm of the target, MuJoCo's own acceptance bound.
    import numpy as np

    dist_xy = float(np.linalg.norm(env.cube_pos[:2] - env.target_pos[:2]))
    assert dist_xy < 0.04, f"cube ended {dist_xy:.3f}m from the target"

    # Release the gripper so a final held-cube slip doesn't leave a false
    # impression; not part of the pass/fail criteria above.
    env.step(
        Action(joint_position=obs.joint_state.position[:5], gripper=GripperCommand(1.0))
    )


def test_reset_puts_the_cube_back_after_an_episode(env):
    """`reset()` must restore the cube: the previous test just carried it to
    the target, and a second episode starting there would be meaningless."""
    import numpy as np

    env.reset(seed=0)
    assert env.cube_pos[:2] == pytest.approx(np.array([0.20, 0.08]), abs=0.005)
    assert env.cube_pos[2] == pytest.approx(env.rest_z, abs=0.005)
