import numpy as np
import pytest


def test_the_builtin_robots_are_registered_with_what_they_own():
    from physai import sim
    from physai.robots import (
        DirectMuJoCoAdapter,
        available_robots,
        available_ros2_robots,
        create_jog_resolver,
        create_robot,
        navigate,
    )
    from physai.robots.so101 import EnvConfig, SO101Env
    from physai.robots.turtlebot import TurtleBot4Env
    from physai.sim import MuJoCoSimulationCore

    assert "so101" in available_robots()
    env = create_robot("so101", render=False)
    try:
        assert isinstance(env, DirectMuJoCoAdapter)
        assert env.robot_spec.name == "so101"
        assert env.robot_spec.kind == "fixed_base_manipulator"
    finally:
        env.close()

    assert available_ros2_robots() == ("so101", "turtlebot4")
    with pytest.raises(ValueError, match="so101"):
        create_robot("does-not-exist")  # the error lists what is available
    with pytest.raises(ValueError, match="has no registered navigation baseline"):
        navigate("so101", goal_x=1.0, goal_y=0.0)
    assert issubclass(TurtleBot4Env, MuJoCoSimulationCore)

    # the SO-101 environment belongs to its robot package and knows no task
    own = SO101Env(EnvConfig(render=False))
    try:
        assert SO101Env.__module__ == "physai.robots.so101.env"
        assert not hasattr(own, "task")
    finally:
        own.close()
    assert not hasattr(sim, "SO101Env")

    class Arm:
        def resolve_twist_jog(self):
            return "resolved"

    assert create_jog_resolver("so101", Arm())() == "resolved"
    assert create_jog_resolver("turtlebot4", object()) is None
    assert create_jog_resolver("_no_such_robot", object()) is None


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


def test_robot_spec_validates_actions_and_unit_declarations():
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

    unitless = RobotSpec(name="unitless", kind="mobile_base", action_modes=("twist",))
    assert unitless.units == {
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


def test_the_scripted_baselines_are_available():
    from physai.planner import ScriptedPlanner
    from physai.tasks import available_tasks, create_task

    plan = ScriptedPlanner((0.2, 0.08, 0.036), (0.2, -0.1, 0.021)).plan("", None)
    assert [subgoal.skill for subgoal in plan.subgoals]
    assert "pick_place" in available_tasks()
    assert create_task("pick_place").name == "pick_place"


def test_registering_an_embodiment_wires_every_factory_and_can_be_extended_once():
    from physai.robots.registry import (
        RobotDescriptor,
        available_robots,
        available_ros2_robots,
        create_env_config,
        create_robot,
        navigate,
        register_embodiment,
        register_env_config,
        register_robot,
        register_ros2_node,
        robot_kind,
        scene_defaults,
    )

    calls: list[str] = []
    register_embodiment(
        "_fake_test_embodiment",
        RobotDescriptor(
            factory=lambda **_: calls.append("factory") or object(),
            kind="fake_kind",
            scene_defaults=lambda: {"fake": True},
            env_config=lambda **_: calls.append("env_config") or object(),
            ros2_node=object,
            navigation=lambda **_: calls.append("navigation") or "ok",
        ),
    )

    assert "_fake_test_embodiment" in available_robots()
    assert robot_kind("_fake_test_embodiment") == "fake_kind"
    assert scene_defaults("_fake_test_embodiment") == {"fake": True}
    assert "_fake_test_embodiment" in available_ros2_robots()
    create_env_config("_fake_test_embodiment")
    navigate("_fake_test_embodiment")
    create_robot("_fake_test_embodiment")
    assert calls == ["env_config", "navigation", "factory"]

    # the piecemeal functions fill one still-empty field of a registered robot
    register_robot("_fake_piecemeal", lambda **_: object(), kind="piece")
    register_env_config("_fake_piecemeal", lambda **kwargs: kwargs)
    register_ros2_node("_fake_piecemeal", object)

    assert robot_kind("_fake_piecemeal") == "piece"
    assert create_env_config("_fake_piecemeal", seed=1) == {"seed": 1}
    assert "_fake_piecemeal" in available_ros2_robots()
    with pytest.raises(ValueError, match="environment config .* already registered"):
        register_env_config("_fake_piecemeal", lambda **_: None)
    with pytest.raises(ValueError, match="not registered"):
        register_env_config("_fake_missing", lambda **_: None)
    with pytest.raises(ValueError, match="already registered"):
        register_robot("_fake_piecemeal", lambda **_: object())
