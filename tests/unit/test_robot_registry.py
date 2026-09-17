import numpy as np
import pytest


def test_so101_is_registered():
    from physai.robots import DirectMuJoCoAdapter, available_robots, create_robot

    assert "so101" in available_robots()
    env = create_robot("so101", render=False)
    try:
        assert isinstance(env, DirectMuJoCoAdapter)
        assert env.robot_spec.name == "so101"
        assert env.robot_spec.kind == "fixed_base_manipulator"
    finally:
        env.close()


def test_ros2_nodes_are_registered_for_supported_robots():
    from physai.robots import available_ros2_robots

    assert available_ros2_robots() == ("so101", "turtlebot4")


def test_so101_environment_is_owned_by_robot_package():
    from physai import sim
    from physai.robots.so101 import EnvConfig, SO101Env

    env = SO101Env(EnvConfig(render=False))
    try:
        assert SO101Env.__module__ == "physai.robots.so101.env"
        assert not hasattr(env, "task")
    finally:
        env.close()
    assert not hasattr(sim, "SO101Env")


def test_turtlebot4_is_registered_and_uses_twist_control():
    from physai.contracts import Action, Twist, Vector3
    from physai.robots import create_robot

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


def test_navigation_capability_is_resolved_by_robot_registry():
    from physai.robots import navigate

    with pytest.raises(ValueError, match="has no registered navigation baseline"):
        navigate("so101", goal_x=1.0, goal_y=0.0)


def test_mujoco_robot_environments_share_simulation_lifecycle():
    from physai.robots.turtlebot import TurtleBot4Env
    from physai.sim import MuJoCoSimulationCore

    assert issubclass(TurtleBot4Env, MuJoCoSimulationCore)


def test_unknown_robot_lists_available_robots():
    from physai.robots import create_robot

    with pytest.raises(ValueError, match="so101"):
        create_robot("does-not-exist")


def test_robot_spec_validates_action_mode_shape_and_values():
    from physai.contracts import Action, Twist
    from physai.robots import RobotSpec

    spec = RobotSpec(
        name="test_arm",
        kind="manipulator",
        action_joint_names=("joint_a", "joint_b"),
        action_modes=("joint_position",),
        units={"joint_position": "rad", "joint_velocity": "rad/s"},
    )
    with pytest.raises(ValueError, match="expects 2 joint targets"):
        spec.validate_action(Action(joint_position=np.zeros(1)))
    with pytest.raises(ValueError, match="does not support action mode 'twist'"):
        spec.validate_action(Action(ee_twist=Twist()))
    with pytest.raises(ValueError, match="non-finite"):
        spec.validate_action(Action(joint_position=np.array([0.0, np.nan])))


def test_robot_spec_exposes_and_validates_si_unit_declarations():
    from physai.robots import RobotSpec

    spec = RobotSpec(name="unitless", kind="mobile_base", action_modes=("twist",))
    assert spec.units == {
        "joint_position": "rad",
        "joint_velocity": "rad/s",
        "linear_velocity": "m/s",
        "angular_velocity": "rad/s",
    }
    with pytest.raises(ValueError, match="invalid unit declarations"):
        RobotSpec(
            name="wrong_units",
            kind="mobile_base",
            action_modes=("twist",),
            units={
                "joint_position": "degrees",
                "joint_velocity": "rad/s",
                "linear_velocity": "m/s",
                "angular_velocity": "rad/s",
            },
        )


def test_builtin_planners_are_discoverable():
    from physai.planner import available_planners

    assert {"scripted", "smolvlm", "claude"} <= set(available_planners())


def test_pick_place_task_is_discoverable():
    from physai.tasks import available_tasks, create_task

    assert "pick_place" in available_tasks()
    assert create_task("pick_place").name == "pick_place"
