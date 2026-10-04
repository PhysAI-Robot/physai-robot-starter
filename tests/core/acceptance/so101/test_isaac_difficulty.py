"""Isaac difficulty knobs: lighting scale and camera position jitter.

Own module because one `SimulationApp` per process is a hard Isaac Sim
constraint (see `test_isaac_env.py`).
"""

from pathlib import Path

import numpy as np
import pytest
from tests.conftest import requires_assets

_isaacsim = pytest.importorskip(
    "isaacsim", reason="uv sync --extra isaac installs isaacsim"
)
if not hasattr(_isaacsim, "SimulationApp"):
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
JITTER = 0.02
LIGHTING = 0.5


@pytest.fixture(scope="module")
def env():
    from physai.robots.so101.isaac_env import IsaacEnvConfig, SO101IsaacEnv
    from physai.sim.mujoco import SingleCubeFixedPlaceSceneConfig

    instance = SO101IsaacEnv(
        IsaacEnvConfig(
            scene=SingleCubeFixedPlaceSceneConfig(),
            usd_out_dir=REPO_ROOT / ".isaac_cache" / "so101_test",
            headless=True,
            render=True,
            cameras=("front", "wrist"),
            lighting_scale=LIGHTING,
            camera_position_jitter=JITTER,
        )
    )
    try:
        yield instance
    finally:
        instance.close()


def camera_translation(env, name: str) -> np.ndarray:
    from pxr import UsdGeom

    prim = env.stage.GetPrimAtPath(env._camera_prims[name])
    return np.array(UsdGeom.Xformable(prim).GetOrderedXformOps()[0].Get())


def test_the_lights_are_scaled_by_the_lighting_level(env):
    from physai.sim.studio import ISAAC_DOME_INTENSITY, ISAAC_KEY_INTENSITY

    dome = env.stage.GetPrimAtPath("/World_lights/dome")
    key = env.stage.GetPrimAtPath("/World_lights/key")
    assert dome.GetAttribute("inputs:intensity").Get() == pytest.approx(
        ISAAC_DOME_INTENSITY * LIGHTING
    )
    assert key.GetAttribute("inputs:intensity").Get() == pytest.approx(
        ISAAC_KEY_INTENSITY * LIGHTING
    )


def test_a_camera_moves_within_the_jitter_but_its_calibration_stays_nominal(env):
    shifts = {}
    nominal_calibration = {}
    for seed in (0, 1):
        env.reset(seed=seed)
        for name in ("front", "wrist"):
            shifts.setdefault(name, []).append(camera_translation(env, name))
    # The calibration is the policy's belief: it must not follow the move.
    front = env.cfg.front_camera
    _, extrinsics = env.camera_calibration("front")
    np.testing.assert_allclose(
        extrinsics.position.as_array(), front.position, atol=1e-9
    )
    nominal_calibration["front"] = np.asarray(front.position)

    for name, (first, second) in shifts.items():
        # Different seeds draw different offsets ...
        assert np.linalg.norm(first - second) > 1e-4
    # ... and each stays inside +/- JITTER per axis of the front camera mount.
    for position in shifts["front"]:
        assert np.max(np.abs(position - nominal_calibration["front"])) <= JITTER + 1e-9


def test_the_same_seed_gives_the_same_camera_offset(env):
    env.reset(seed=7)
    first = camera_translation(env, "front")
    env.reset(seed=3)
    env.reset(seed=7)
    np.testing.assert_allclose(camera_translation(env, "front"), first)
