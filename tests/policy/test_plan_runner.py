import pytest
from conftest import requires_assets

pytestmark = pytest.mark.assets


@requires_assets
def test_plan_runner_executes_a_scripted_plan(env):
    from physai.planner import ScriptedPlanner
    from physai.policy.plan_runner import PlanRunner

    obs = env.reset(seed=0)
    plan = ScriptedPlanner(env.cube_pos, env.target_pos).plan("test", obs)
    runner = PlanRunner(env.kin, plan, dt=env.control_dt)
    runner.reset(obs)
    for _ in range(env.cfg.max_steps):
        obs, *_rest = env.step(runner.act(obs))
        runner.note_progress(
            env.kin.pinch_center(env.data),
            env.joint_to_gripper(obs.joint_state.position[5]),
        )
        if runner.done:
            break
    assert runner.index > 0, "plan runner never completed a sub-goal"
