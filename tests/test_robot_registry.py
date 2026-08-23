def test_so101_is_registered():
    from physai.robots import available_robots, create_robot

    assert "so101" in available_robots()
    env = create_robot("so101", render=False)
    try:
        assert env.robot_spec.name == "so101"
        assert env.robot_spec.kind == "fixed_base_manipulator"
    finally:
        env.close()


def test_turtlebot4_is_registered_and_uses_twist_control():
    import numpy as np

    from physai.contracts import Action, Twist, Vector3
    from physai.robots import available_robots, create_robot

    assert "turtlebot4" in available_robots()
    env = create_robot("turtlebot4", control_hz=10.0)
    try:
        obs = env.reset()
        assert env.robot_spec.name == "turtlebot4"
        assert env.robot_spec.kind == "mobile_base"
        assert env.robot_spec.supports("base_velocity", "odometry")
        assert not env.robot_spec.supports("arm_kinematics")
        assert env.robot_spec.action_modes == ("twist",)
        obs, _, _, _, _ = env.step(Action(ee_twist=Twist(linear=Vector3(x=0.2))))
        assert obs.step == 1
        assert env.model.nu == 3
        assert obs.ee_pose.pose.position.z > 0.0
        assert np.all(obs.joint_state.position > 0.0)
        assert np.all(obs.joint_state.velocity > 0.0)
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