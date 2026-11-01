from types import SimpleNamespace

import numpy as np
import pytest

from physai.planner import StackPlanner
from physai.tasks import create_task

HALF = 0.014
TABLE = 0.02
PAD = (0.20, -0.10, 0.021)
REST = TABLE + HALF


def _backend(**cubes):
    return SimpleNamespace(
        cube_positions={k: np.array(v) for k, v in cubes.items()},
        target_pos=np.array(PAD),
        ee_pos=np.zeros(3),
        table_top=TABLE,
        cube_half=HALF,
    )


def test_a_stack_in_order_is_at_target():
    task = create_task("stack_cubes", order=("red", "blue"))
    info = task.evaluate(
        _backend(
            red=(0.20, -0.10, REST),
            blue=(0.20, -0.10, REST + 2 * HALF),
            yellow=(0.25, 0.0, REST),
        )
    )
    assert info["cubes_stacked"] == 2 and info["at_target"]


def test_levels_count_from_the_bottom_so_a_gap_ends_the_count():
    task = create_task("stack_cubes", order=("red", "blue"))
    info = task.evaluate(
        _backend(red=(0.30, 0.0, REST), blue=(0.20, -0.10, REST + 2 * HALF))
    )
    assert info["cubes_stacked"] == 0 and not info["at_target"]


def test_a_cube_held_above_its_level_is_not_stacked():
    task = create_task("stack_cubes", order=("red", "blue"))
    info = task.evaluate(
        _backend(red=(0.20, -0.10, REST), blue=(0.20, -0.10, REST + 2 * HALF + 0.015))
    )
    assert info["cubes_stacked"] == 1 and not info["at_target"]


def test_a_cube_off_the_table_is_dropped():
    task = create_task("stack_cubes", order=("red",))
    info = task.evaluate(_backend(red=(0.20, -0.10, TABLE - 0.1)))
    assert info["cube_dropped"] and task.terminated(None, info)


def test_the_order_must_name_each_cube_once():
    with pytest.raises(ValueError):
        create_task("stack_cubes", order=("red", "red"))


def _plan(order=("red", "blue"), **changes):
    import dataclasses

    from physai.robots.so101.mujoco_env import SO101_GRASP

    profile = dataclasses.replace(SO101_GRASP, **changes)
    cubes = {
        "red": np.array([0.18, 0.02, 0.034]),
        "blue": np.array([0.19, 0.07, 0.034]),
        "yellow": np.array([0.18, 0.12, 0.034]),
    }
    planner = StackPlanner(cubes, PAD, order, HALF, TABLE, profile)
    return planner.plan("stack", SimpleNamespace(ee_pose=None)), profile


def _top_down(order=("red", "blue")):
    """The plan of an arm whose top-down grasp reaches the tower, as the ViperX's does."""
    return _plan(order, place_tilt=None)


def test_the_plan_picks_each_cube_in_order_and_names_where_it_goes():
    plan, _ = _top_down()
    graspers = [g.target_description for g in plan.subgoals if g.skill == "grasp"]
    assert graspers == ["red cube", "blue cube"]
    releases = [g.target_description for g in plan.subgoals if g.skill == "release"]
    assert releases == ["pad", "on red cube"]


def test_each_level_is_one_cube_higher_and_carried_clear_of_the_tower():
    plan, p = _top_down()
    zs = {}
    for g in plan.subgoals:
        if g.skill == "release":
            zs[g.target_description] = g.waypoint.pose.position.as_array()[2]
    assert np.isclose(zs["on red cube"] - zs["pad"], 2 * HALF)
    lifts = [g for g in plan.subgoals if g.skill == "lift"]
    tower_top = TABLE + 2 * HALF  # one cube stands on the pad when blue is carried
    assert lifts[1].waypoint.pose.position.as_array()[2] - p.grasp_dz - HALF > tower_top


def test_no_retreat_follows_the_last_cube_of_a_top_down_stack():
    plan, _ = _top_down()
    assert plan.subgoals[-1].skill == "release"


def _orientations(plan):
    from physai.planner.base import approach_from_orientation

    out = []
    for g in plan.subgoals:
        xy = g.waypoint.pose.position.as_array()[:2]
        d = approach_from_orientation(g.waypoint.pose.orientation, xy)
        out.append(None if d is None else float(np.degrees(np.arcsin(-d[2]))))
    return out


def test_a_tall_tower_is_put_down_from_the_side_with_the_wrist_turned_in_small_steps():
    plan, p = _plan(("red", "blue", "yellow"))
    tilts = _orientations(plan)
    assert tilts[0] is None  # the pick is top-down
    placed = [
        t
        for g, t in zip(plan.subgoals, tilts)
        if g.skill == "release"  # let go with the gripper flat
    ]
    assert len(placed) == 3 and all(abs(t) < 1e-6 for t in placed)
    steps = [
        abs(b - a) for a, b in zip(tilts, tilts[1:]) if a is not None and b is not None
    ]
    assert max(steps) <= 15.0 + 1e-6, "one jump between two poses sags the pinch point"


def test_the_flat_phase_rolls_the_jaws_so_the_put_down_cube_is_not_left_on_one():
    from physai.planner.base import gripper_quat_wxyz

    plan, p = _plan(("red",))
    release = next(g for g in plan.subgoals if g.skill == "release")
    xy = release.waypoint.pose.position.as_array()[:2]
    q = gripper_quat_wxyz(release.waypoint.pose.orientation, xy)
    w, x, y, z = q
    # The jaws close along the site's z axis; at the flat it is sideways, not vertical.
    closing = np.array(
        [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)]
    )
    assert abs(closing[2]) < 1e-6 and p.place_roll == 90.0


def test_every_level_is_carried_over_the_tower_and_loose_cubes():
    plan, p = _plan(("red", "blue", "yellow"))
    carries = [
        g.waypoint.pose.position.as_array()[2]
        for g in plan.subgoals
        if g.skill == "move_above" and not g.target_description
    ]
    loose_cube_top = TABLE + 2 * HALF
    assert min(carries) > loose_cube_top + HALF + p.grasp_dz
