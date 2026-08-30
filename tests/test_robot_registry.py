class RecordingTransport:
    def __init__(self):
        self.messages = []
        self.closed = False

    def publish(self, topic, message):
        self.messages.append((topic, message))

    def subscribe(self, topic, callback):
        del topic, callback

    def close(self):
        self.closed = True


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


def test_so101_environment_is_owned_by_robot_package():
    from physai.robots.so101 import SO101Env
    import physai.sim as sim

    assert SO101Env.__module__ == "physai.robots.so101.env"
    assert not hasattr(sim, "SO101Env")


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


def test_robot_spec_validates_action_mode_shape_and_values():
    import numpy as np
    import pytest

    from physai.contracts import Action, Twist
    from physai.robots import RobotSpec

    spec = RobotSpec(
        name="test_arm",
        kind="manipulator",
        action_joint_names=("joint_a", "joint_b"),
        action_modes=("joint_position",),
    )
    with pytest.raises(ValueError, match="expects 2 joint targets"):
        spec.validate_action(Action(joint_position=np.zeros(1)))
    with pytest.raises(ValueError, match="does not support action mode 'twist'"):
        spec.validate_action(Action(ee_twist=Twist()))
    with pytest.raises(ValueError, match="non-finite"):
        spec.validate_action(Action(joint_position=np.array([0.0, np.nan])))


def test_so101_ros2_mujoco_adapter_publishes_contract_topics():
    from physai.robots import create_robot

    transport = RecordingTransport()
    env = create_robot("so101", adapter="ros2_mujoco", transport=transport,
                       render=False)
    try:
        env.reset(seed=0)
        topics = [topic for topic, _ in transport.messages]
        assert "/joint_states" in topics
        assert "/camera/front/image_raw" not in topics
        assert env.robot_spec.name == "so101"
    finally:
        env.close()
    assert transport.closed


def test_ros2_adapter_requires_transport():
    import pytest

    from physai.robots import create_robot

    with pytest.raises(ValueError, match="requires a ROS2 transport"):
        create_robot("so101", adapter="ros2_mujoco", render=False)


def test_builtin_planners_are_discoverable():
    from physai.planner import available_planners

    assert {"scripted", "smolvlm", "claude"} <= set(available_planners())


def test_pick_place_task_is_discoverable():
    from physai.tasks import available_tasks, create_task

    assert "pick_place" in available_tasks()
    assert create_task("pick_place").name == "pick_place"