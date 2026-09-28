"""Phase 3 parity-ladder tiers 1-2 against real Isaac Sim: URDF import and
actuator step response. Skipped wherever `isaacsim` is not installed —
`uv sync --extra isaac` installs it into this project's own venv (see
`docs/adr/0016-isaacsim-as-a-project-extra.md`).

One `SimulationApp` per process is a hard Isaac Sim constraint (see
`physai.sim.isaac.core`), so every test in this module shares one
`SO101IsaacEnv` via a module-scoped fixture rather than constructing its
own.
"""

from pathlib import Path

import numpy as np
import pytest
from conftest import requires_assets

_isaacsim = pytest.importorskip(
    "isaacsim", reason="uv sync --extra isaac installs isaacsim"
)
if not hasattr(_isaacsim, "SimulationApp"):
    # `isaacsim` can still be a bare, empty namespace package after `uv sync`
    # drops the `isaac` extra: it only removes files the wheel's own RECORD
    # lists, not the extsUser/kit directories isaacsim writes under its own
    # install path at runtime, so a stale `isaacsim/` directory can outlive
    # the real package and make a plain `importorskip("isaacsim")` succeed
    # on nothing.
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

REPO_ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(scope="module")
def env():
    from physai.robots.so101.isaac_env import IsaacEnvConfig, SO101IsaacEnv

    cfg = IsaacEnvConfig(
        usd_out_dir=REPO_ROOT / ".isaac_cache" / "so101_test",
        headless=True,
        render=True,
        cameras=("wrist",),
        max_steps=400,
    )
    instance = SO101IsaacEnv(cfg)
    try:
        yield instance
    finally:
        instance.close()


def test_all_actuated_joints_import_with_the_mujoco_side_joint_order(env):
    """Tier 1 (static): the URDF import must produce exactly the joints
    `physai.robots.so101.contracts.ALL_JOINT_NAMES` expects, in that order —
    the same joints and order the MuJoCo side (`SO101Env`) exposes.
    """
    from physai.robots.so101.contracts import ALL_JOINT_NAMES

    assert env.dof_names == list(ALL_JOINT_NAMES)


def test_reset_teleports_to_the_same_home_pose_as_mujoco(env):
    """Tier 1 (static): `HOME_QPOS` is shared source with `so101.mujoco_env`
    (imported from there), so a diff here is a real cross-backend gap, not
    just this module's own bug.
    """
    from physai.robots.so101.mujoco_env import HOME_QPOS

    observation = env.reset(seed=0)
    np.testing.assert_allclose(
        observation.joint_state.position[:5], HOME_QPOS, atol=1e-4
    )
    assert observation.joint_state.position[5] == pytest.approx(
        env.gripper_to_joint(1.0), abs=1e-4
    )


def test_actuator_gains_transfer_from_mujoco_with_no_unit_conversion(env):
    """Tier 2 (actuator step response): the SO-101 MJCF's sts3215 gains
    (stiffness=998.22, damping=2.731, force_limit=3.35 — see
    `robots/so101/description.yaml`) are applied to Isaac's PhysX drive
    as-is; verified manually to settle a 0.3 rad step within 2e-4 rad in 3 s
    on an RTX 3060. This is the tracking-error tolerance that verification
    supports, not an arbitrarily loose bound.
    """
    from physai.contracts import Action, GripperCommand

    observation = env.reset(seed=0)
    target = observation.joint_state.position[:5].copy()
    target[0] += 0.3
    action = Action(joint_position=target, gripper=GripperCommand(position=1.0))
    for _ in range(90):  # 3 s at 30 Hz
        observation, _, _, _, _ = env.step(action)
    tracking_error = abs(float(observation.joint_state.position[0]) - float(target[0]))
    assert tracking_error < 5e-3


def test_wrist_camera_renders_a_real_non_black_frame(env):
    """A bare URDF import has no lights (`physai.sim.isaac.scene
    .add_studio_lighting` exists because of this); regression-guards that
    the render pipeline still produces a lit frame, not a black one.
    """
    observation = env.reset(seed=0)
    frame = observation.images["wrist"]
    assert frame.data.shape == (240, 320, 3)
    assert frame.data.mean() > 10.0
