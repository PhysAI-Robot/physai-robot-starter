import pytest
from tests.conftest import requires_assets

pytestmark = [pytest.mark.acceptance, pytest.mark.assets, pytest.mark.slow]


@requires_assets
def test_visual_servo_single_cube_fixed_place_settles_from_multiple_seeds():
    from physai.robots.so101 import EnvConfig, SO101Env
    from physai.sim.mujoco import SingleCubeFixedPlaceSceneConfig
    from physai.tasks import TaskRuntime, create_task
    from research.classical_control.so101_visual_servo import SO101VisualServoPolicy

    robot = SO101Env(
        EnvConfig(
            scene=SingleCubeFixedPlaceSceneConfig(),
            seed=0,
            render=True,
            max_steps=400,
        )
    )
    env = TaskRuntime(robot, create_task("single_cube_fixed_place"))
    try:
        policy = SO101VisualServoPolicy(env)
        for seed in (0, 1, 2):
            observation = env.reset(seed=seed)
            policy.reset(observation)
            info = {}
            for _ in range(400):
                observation, _, terminated, truncated, info = env.step(
                    policy.act(observation)
                )
                if terminated or truncated:
                    break

            assert info["success"], f"visual servo failed for seed {seed}: {info}"
            assert info["dist_cube_target"] < 0.04
            assert policy.metrics.phase in {"RELEASE", "RETREAT", "DONE"}
            assert policy.metrics.ee_error_m <= 0.012
            assert policy.metrics.settling_time_s is not None
            assert policy.metrics.failure_reason is None
    finally:
        env.close()
