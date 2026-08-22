def test_so101_is_registered():
    from physai.robots import available_robots, create_robot

    assert "so101" in available_robots()
    env = create_robot("so101", render=False)
    try:
        assert env.robot_spec.name == "so101"
        assert env.robot_spec.kind == "fixed_base_manipulator"
    finally:
        env.close()


def test_unknown_robot_lists_available_robots():
    import pytest

    from physai.robots import create_robot

    with pytest.raises(ValueError, match="so101"):
        create_robot("does-not-exist")


def test_builtin_planners_are_discoverable():
    from physai.planner import available_planners

    assert {"scripted", "smolvlm", "claude"} <= set(available_planners())


def test_pick_place_task_is_discoverable():
    from physai.tasks import available_tasks, create_task

    assert "pick_place" in available_tasks()
    assert create_task("pick_place").name == "pick_place"