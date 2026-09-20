import numpy as np
import pytest
from conftest import requires_assets

pytestmark = [pytest.mark.acceptance, pytest.mark.assets, pytest.mark.slow]


@requires_assets
def test_both_observation_cameras_carry_signal():
    from physai.robots.so101 import EnvConfig, SO101Env
    from physai.sim import PickPlaceMinimalSceneConfig

    robot = SO101Env(
        EnvConfig(
            scene=PickPlaceMinimalSceneConfig(camera_width=128, camera_height=128),
            seed=0,
            render=True,
            max_steps=200,
        )
    )
    try:
        robot.reset(seed=0)
        for name in ("front", "wrist"):
            frame = robot.render_camera(name).astype(float)
            assert frame.std() > 20.0, f"{name} camera is nearly uniform"
    finally:
        robot.close()


@requires_assets
def test_wrist_camera_looks_toward_the_object_it_is_grasping():
    import mujoco

    from physai.robots.so101 import EnvConfig, SO101Env
    from physai.tasks import TaskRuntime, create_task

    robot = SO101Env(EnvConfig(seed=0, render=False, max_steps=200))
    env = TaskRuntime(robot, create_task("pick_place"))
    try:
        env.reset(seed=0)
        cam = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
        view = -robot.data.cam_xmat[cam].reshape(3, 3)[:, 2]
        to_cube = env.cube_pos - robot.data.cam_xpos[cam]
        to_cube /= np.linalg.norm(to_cube)
        assert float(view @ to_cube) > 0.5
    finally:
        env.close()
