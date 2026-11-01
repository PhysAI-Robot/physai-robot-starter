from pathlib import Path

import pytest
from tests.conftest import requires_viperx300s_assets

MANIFEST = Path("configs/manifests/viperx300s_single_cube_fixed_place.yaml")


@requires_viperx300s_assets
@pytest.mark.parametrize("seed", [3, 4, 5])
def test_scripted_plan_with_the_viperx300s_profile_places_the_cube(seed):
    from physai.config import load_manifest
    from physai.config.compat import with_overrides
    from physai.planner import ScriptedPlanner
    from physai.policy.plan_runner import PlanRunner
    from physai.runtime import create_session

    manifest = with_overrides(
        load_manifest(MANIFEST), seed=seed, policy="idle", max_steps=800
    )
    session = create_session(manifest, render=False)
    try:
        env = session.runtime.robot
        obs = env.reset(seed=seed)
        profile = env.grasp_profile()
        plan = ScriptedPlanner(env.cube_pos, env.target_pos, profile).plan("", obs)
        runner = PlanRunner(
            env.kin,
            plan,
            dt=env.control_dt,
            approach_dir=profile.approach_dir,
            ik_kwargs=profile.ik_kwargs,
        )
        runner.reset(obs)
        joints = len(env.kin.joint_names)
        info = {}
        for _ in range(800):
            obs, _, terminated, truncated, info = env.step(runner.act(obs))
            runner.note_progress(
                env.pinch_center(),
                env.joint_to_gripper(obs.joint_state.position[joints]),
            )
            if runner.done or terminated or truncated:
                break
        assert info["success"], info["dist_cube_target"]
    finally:
        session.close()
