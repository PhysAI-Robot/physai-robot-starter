import numpy as np

from physai.contracts import Observation
from physai.planner import ScriptedPlanner

PICK, PLACE = (0.30, 0.05, 0.034), (0.32, -0.10, 0.021)


def _plan(profile=None):
    planner = ScriptedPlanner(PICK, PLACE, profile)
    return planner.plan("pick and place", Observation(joint_state=None, images={}))


def _steps(plan):
    return [
        (g.skill, g.waypoint.pose.position.as_array().round(4).tolist(), g.gripper)
        for g in plan.subgoals
    ]


def test_without_a_profile_the_plan_is_the_four_step_default():
    steps = _steps(_plan())
    assert [skill for skill, _, _ in steps] == [
        "move_above",
        "grasp",
        "move_above",
        "release",
    ]
    assert steps[1][1] == list(PICK)


def test_the_so101_profile_descends_open_with_the_wrist_top_down():
    from physai.robots.so101.mujoco_env import SO101_GRASP as p

    plan = _plan(p)
    assert [g.skill for g in plan.subgoals] == [
        "move_above",
        "descend",
        "grasp",
        "move_above",
        "move_above",
        "release",
    ]
    assert [g.gripper for g in plan.subgoals][:3] == [
        p.grip_open,
        p.grip_open,
        p.grip_hold,
    ]
    assert p.approach_dir == (0.0, 0.0, -1.0)


def test_the_viperx300s_profile_descends_open_lifts_and_holds_the_pinch_above_the_cube():
    from physai.robots.viperx300s.mujoco_env import VIPERX300S_GRASP as p

    plan = _plan(p)
    assert [g.skill for g in plan.subgoals] == [
        "move_above",
        "descend",
        "grasp",
        "lift",
        "move_above",
        "move_above",
        "release",
    ]
    grips = [g.gripper for g in plan.subgoals]
    assert grips == [
        p.grip_open,
        p.grip_open,
        p.grip_hold,
        p.grip_hold,
        p.grip_hold,
        p.grip_hold,
        p.grip_open,
    ]
    z = [g.waypoint.pose.position.as_array()[2] for g in plan.subgoals]
    assert np.isclose(z[2], PICK[2] + p.grasp_dz)  # the grasp
    assert np.isclose(z[1], z[2])  # reached open, then closed
    assert np.isclose(z[0], z[2] + p.hover)
    assert np.isclose(z[3], z[2] + p.carry)
    assert np.isclose(z[5], PLACE[2] + p.release_dz)  # lowered, still holding
    assert np.isclose(z[6], z[5])  # then opened
