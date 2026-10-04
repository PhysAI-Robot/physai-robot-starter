"""Isaac Sim places the cube where MuJoCo does for the same seed, and a task
layer reads Isaac's state. Skipped wherever `isaacsim` is not installed.

One `SimulationApp` per process is a hard Isaac Sim constraint, so this module
shares one env through a module-scoped fixture and, like the other Isaac
modules, is meant to be run on its own.
"""

import json
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
GOLDEN = json.loads(
    (Path(__file__).with_name("golden_layouts.json")).read_text(encoding="utf-8")
)["single_default"]


@pytest.fixture(scope="module")
def env():
    from physai.robots.so101.isaac_env import IsaacEnvConfig, SO101IsaacEnv
    from physai.sim.mujoco import SingleCubeFixedPlaceSceneConfig

    cfg = IsaacEnvConfig(
        scene=SingleCubeFixedPlaceSceneConfig(),
        usd_out_dir=REPO_ROOT / ".isaac_cache" / "so101_layout_test",
        headless=True,
        render=True,
        cameras=("front", "wrist"),
        max_steps=50,
    )
    instance = SO101IsaacEnv(cfg)
    try:
        yield instance
    finally:
        instance.close()


def test_a_seed_places_the_cube_where_mujoco_does(env):
    for seed, expected in GOLDEN.items():
        env.reset(seed=int(seed))
        cube = env.cube_pos
        # The cube spawns 2 mm above the table, as in MuJoCo, and settles onto
        # it (sliding a couple of mm) during the reset's hold step.
        assert cube[:2] == pytest.approx(expected["cube"][:2], abs=4e-3), seed
        assert cube[2] == pytest.approx(expected["cube"][2], abs=3e-3), seed
        assert env.target_pos == pytest.approx(expected["target_pos"], abs=1e-6)


def test_the_front_camera_sees_the_cube_where_it_is(env):
    """The cube must render saturated red, not washed-out pink: with Isaac's
    default tone mapping and sRGB encoding its top face lost most of its
    saturation, `ColorBlobDetector` found a fragment, and the triangulated
    position was off by centimetres. The blob has to sit on the cube's true
    projection and be about as large as the cube looks."""
    from physai.contracts import Action
    from physai.robots.so101.isaac_env import HOME_QPOS
    from research.classical_control.so101_visual_servo import ColorBlobDetector

    obs = env.reset(seed=0)
    hold = Action(joint_position=HOME_QPOS.copy())
    for _ in range(4):  # the render catches up with the teleport one step later
        obs, *_ = env.step(hold)
    frame = obs.images["front"]
    intrinsics, extrinsics = frame.intrinsics, frame.extrinsics
    rotation = extrinsics.orientation.to_matrix()
    in_camera = rotation.T @ (env.cube_pos - extrinsics.position.as_array())
    true_pixel = np.array(
        [
            intrinsics.fx * in_camera[0] / in_camera[2] + intrinsics.cx,
            intrinsics.fy * in_camera[1] / in_camera[2] + intrinsics.cy,
        ]
    )

    feature = ColorBlobDetector().detect(frame)

    assert feature is not None
    assert np.linalg.norm(feature.pixel - true_pixel) < 2.0
    assert feature.area > 250


def test_the_task_layer_reads_isaacs_state(env):
    from physai.contracts import Action
    from physai.robots.so101.isaac_env import HOME_QPOS
    from physai.tasks import TaskRuntime, create_task

    runtime = TaskRuntime(env, create_task("single_cube_fixed_place"))
    runtime.reset(seed=0)
    _, _, terminated, _, info = runtime.step(Action(joint_position=HOME_QPOS.copy()))

    expected = np.linalg.norm(env.cube_pos[:2] - env.target_pos[:2])
    assert info["dist_cube_target"] == pytest.approx(expected, abs=1e-6)
    assert info["success"] is False and terminated is False
    assert env.table_top == pytest.approx(0.02) and env.cube_half == 0.014


def test_the_viewer_mirror_holds_the_table_the_target_and_the_cube(env):
    """`--serve` draws `env.model`/`env.data`: it must contain the scene, not
    only the arm, and the mirrored cube must follow Isaac's."""
    import mujoco

    def geom_names(model):
        return {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index)
            for index in range(model.ngeom)
        }

    names = geom_names(env.model)
    assert "target_pad" in names
    assert any(name and "table" in name for name in names)
    assert mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "cube") >= 0

    for seed in (0, 7):
        env.reset(seed=seed)
        env.observe()
        body = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        np.testing.assert_allclose(env.data.xpos[body], env.cube_pos, atol=1e-6)
