import pytest
from conftest import requires_assets

pytestmark = pytest.mark.assets


@requires_assets
def test_expert_state_machine_reaches_the_end(env):
    from research.scripted_experts.so101_pick_place_expert import (
        Phase,
        SO101PickPlaceExpert,
    )

    obs = env.reset(seed=0)
    policy = SO101PickPlaceExpert(env.kin, env)
    policy.reset(obs)
    seen = {policy.phase}
    for _ in range(env.cfg.max_steps):
        obs, *_rest = env.step(policy.act(obs))
        seen.add(policy.phase)
        if policy.done:
            break
    assert Phase.CLOSE in seen and Phase.SQUEEZE in seen and Phase.LIFT in seen


@requires_assets
def test_expert_actually_closes_on_the_cube(env):
    """The grasp must be a real contact, not the jaws shutting on empty air."""
    import mujoco

    from research.scripted_experts.so101_pick_place_expert import (
        Phase,
        SO101PickPlaceExpert,
    )

    obs = env.reset(seed=0)
    policy = SO101PickPlaceExpert(env.kin, env)
    policy.reset(obs)
    pad_contact = False
    for _ in range(env.cfg.max_steps):
        obs, *_rest = env.step(policy.act(obs))
        if policy.phase in (Phase.SQUEEZE, Phase.LIFT):
            for c in range(env.data.ncon):
                names = {
                    mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, g)
                    for g in (env.data.contact[c].geom1, env.data.contact[c].geom2)
                }
                if "cube_geom" in names and names & {"pad_static", "pad_moving"}:
                    pad_contact = True
        if policy.done:
            break
    assert pad_contact, "expert never made pad-to-cube contact"
