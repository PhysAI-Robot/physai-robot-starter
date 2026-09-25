import pytest
from conftest import requires_assets

pytestmark = pytest.mark.assets


def grasp_and_lift(env):
    """Grasp the cube with the scripted expert and stop just after it lifts."""
    from research.scripted_experts.so101_pick_place_expert import SO101PickPlaceExpert

    obs = env.reset(seed=0)
    policy = SO101PickPlaceExpert(env.kin, env)
    policy.reset(obs)
    for _ in range(env.cfg.max_steps):
        obs, *_rest = env.step(policy.act(obs))
        if policy.phase.name == "LIFT":
            break
    for _ in range(20):
        obs, *_rest = env.step(policy.act(obs))
    return obs


def cube_in_gripper(env):
    """Cube centre along the gripper's approach axis, in millimetres."""
    robot = env.robot if hasattr(env, "robot") else env
    data = robot.data
    site_rotation = data.site_xmat[robot.kin.site_id].reshape(3, 3)
    offset = data.geom_xpos[robot.cube_geom_id] - data.site_xpos[robot.kin.site_id]
    return 1000.0 * float((offset @ site_rotation)[0])


@requires_assets
def test_a_held_cube_does_not_creep_out_of_the_fingers(env):
    """MuJoCo's soft friction lets a held object creep out at ~1 mm/s under its
    own weight unless the no-slip solver is on; a cube held with the gripper
    key for ~15 s used to fall out."""
    import numpy as np

    from physai.contracts import Action, GripperCommand

    obs = grasp_and_lift(env)
    hold = np.array(obs.joint_state.position[:5], dtype=np.float64)
    start = cube_in_gripper(env)

    for _ in range(int(12 * env.cfg.control_hz)):
        env.step(Action(joint_position=hold, gripper=GripperCommand(0.0)))

    assert abs(cube_in_gripper(env) - start) < 1.0  # millimetres, over 12 s
    assert env.data.geom_xpos[env.cube_geom_id][2] > 0.05  # still off the table
