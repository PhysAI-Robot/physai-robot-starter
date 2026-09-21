import pytest
from conftest import requires_assets

pytestmark = [pytest.mark.acceptance, pytest.mark.assets, pytest.mark.slow]


@requires_assets
def test_visual_servo_pick_place_settles_from_multiple_seeds():
    from physai.robots.so101 import EnvConfig, SO101Env
    from physai.robots.so101.visual_servo import SO101VisualServoPolicy
    from physai.sim import PickPlaceMinimalSceneConfig
    from physai.tasks import TaskRuntime, create_task

    robot = SO101Env(
        EnvConfig(
            scene=PickPlaceMinimalSceneConfig(camera_width=224, camera_height=224),
            seed=0,
            render=True,
            max_steps=400,
        )
    )
    env = TaskRuntime(robot, create_task("pick_place"))
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
