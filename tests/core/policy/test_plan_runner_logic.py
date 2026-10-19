from types import SimpleNamespace

import numpy as np

from physai.planner.base import Plan, SubGoal
from physai.policy.plan_runner import PlanRunner

AT = np.array([0.2, 0.0, 0.05])


class FakeKin:
    """IK that always lands exactly on its target, so the runner's own logic is on test."""

    joint_names = ("a", "b")

    def ik_pinch(self, target, direction, q_init=None, **_):
        return SimpleNamespace(qpos=np.asarray(q_init, dtype=float))


def _runner(*subgoals, **kwargs):
    runner = PlanRunner(FakeKin(), Plan("", list(subgoals)), **kwargs)
    obs = SimpleNamespace(joint_state=SimpleNamespace(position=np.zeros(3), name=None))
    runner.reset(obs)
    return runner, obs


def _grasp(**kwargs):
    return _runner(SubGoal.from_xyz("grasp", AT, gripper=0.06), **kwargs)


def test_a_gripper_stopped_on_the_cube_counts_as_settled():
    runner, _ = _grasp()
    # Closing, then stopped at 0.2: it never reaches the 0.06 command.
    for grip in [0.5, 0.4, 0.3, 0.2] + [0.2] * 12:
        runner.note_progress(AT, grip)
    assert runner.done


def test_a_gripper_still_moving_is_not_settled():
    runner, _ = _grasp()
    for grip in np.linspace(0.5, 0.2, 16):
        runner.note_progress(AT, float(grip))
    assert not runner.done


def test_a_stopped_gripper_does_not_finish_a_grasp_away_from_the_cube():
    runner, _ = _grasp()
    for _ in range(20):
        runner.note_progress(AT + np.array([0.0, 0.0, 0.1]), 0.2)
    assert not runner.done


def test_the_command_must_catch_up_with_the_ik_solution_before_a_waypoint_counts():
    runner, _ = _runner(SubGoal.from_xyz("move_above", AT, gripper=1.0))
    runner._lag = 0.5  # the wrist is still turning
    runner.note_progress(AT, 1.0)
    assert not runner.done
    runner._lag = 0.0
    runner.note_progress(AT, 1.0)
    assert runner.done


def test_the_cubes_waypoints_follow_the_cube_when_the_approach_begins():
    above = SubGoal.from_xyz(
        "move_above", AT + [0, 0, 0.05], target_description="cube", gripper=1.0
    )
    grasp = SubGoal.from_xyz("grasp", AT, target_description="cube", gripper=0.06)
    pad = SubGoal.from_xyz("move_above", [0.1, 0.1, 0.05], target_description="pad")
    cube = np.array([0.25, 0.03, 0.03])
    runner, obs = _runner(
        above, grasp, pad, locate=lambda name: cube if name == "cube" else None
    )
    runner.act(obs)
    xyz = [g.waypoint.pose.position.as_array() for g in runner.plan.subgoals]
    assert np.allclose(xyz[0], [0.25, 0.03, 0.10])  # moved over the cube, same height
    assert np.allclose(xyz[1], [0.25, 0.03, 0.05])
    assert np.allclose(xyz[2], [0.1, 0.1, 0.05])  # the pad is not the cube
    assert np.allclose(above.waypoint.pose.position.as_array(), AT + [0, 0, 0.05])


class RecordingKin(FakeKin):
    def ik_pinch(self, target, direction, q_init=None, **kwargs):
        self.calls.append((direction, kwargs))
        return super().ik_pinch(target, direction, q_init=q_init, **kwargs)

    def __init__(self):
        self.calls = []


def test_a_tilted_waypoint_sends_the_full_gripper_orientation_to_the_ik():
    from physai.planner.base import approach_orientation

    point = SubGoal.from_xyz("move_above", [0.3, 0.0, 0.1], gripper=1.0)
    point.waypoint.pose.orientation = approach_orientation(
        0.0, 90.0
    )  # flat, jaws sideways
    plain = SubGoal.from_xyz("move_above", [0.3, 0.0, 0.1], gripper=1.0)
    kin = RecordingKin()
    for goal in (point, plain):
        runner = PlanRunner(kin, Plan("", [goal]), approach_dir=(0.0, 0.0, -1.0))
        runner.reset(
            SimpleNamespace(
                joint_state=SimpleNamespace(position=np.zeros(3), name=None)
            )
        )
        runner.act(SimpleNamespace(joint_state=SimpleNamespace(position=np.zeros(3))))
    (flat_dir, flat_kw), (plain_dir, plain_kw) = kin.calls
    assert (
        np.allclose(flat_dir, [1.0, 0.0, 0.0]) and len(flat_kw["target_quat_wxyz"]) == 4
    )
    assert (
        np.allclose(plain_dir, [0.0, 0.0, -1.0]) and "target_quat_wxyz" not in plain_kw
    )


def test_a_tilted_waypoint_points_outward_from_the_base_wherever_it_is():
    from physai.planner.base import approach_from_orientation, approach_orientation

    q = approach_orientation(0.0)
    for xy in ((0.3, 0.0), (0.0, 0.3), (-0.2, 0.2)):
        d = approach_from_orientation(q, xy)
        assert np.allclose(d[:2], np.array(xy) / np.hypot(*xy)) and abs(d[2]) < 1e-9
    assert approach_from_orientation(approach_orientation(90.0), (0.3, 0.0)) is None


def test_retargeting_keeps_the_orientation_of_a_waypoint():
    from physai.planner.base import approach_orientation

    goal = SubGoal.from_xyz("move_above", AT, target_description="cube", gripper=1.0)
    goal.waypoint.pose.orientation = approach_orientation(0.0)
    runner, obs = _runner(goal, locate=lambda name: np.array([0.25, 0.03, 0.03]))
    runner.act(obs)
    moved = runner.plan.subgoals[0].waypoint.pose
    assert np.allclose(moved.position.as_array()[:2], [0.25, 0.03])
    assert moved.orientation == approach_orientation(0.0)
