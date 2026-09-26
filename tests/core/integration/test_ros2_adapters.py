from types import SimpleNamespace

import numpy as np
import pytest

from tests.core.support.fakes import RecordingTransport


@pytest.mark.integration
def test_so101_ros2_mujoco_adapter_publishes_contract_topics():
    from physai.robots import create_robot

    transport = RecordingTransport()
    env = create_robot(
        "so101", adapter="ros2_mujoco", transport=transport, render=False
    )
    try:
        env.reset(seed=0)
        topics = [topic for topic, _ in transport.messages]
        assert "/joint_states" in topics
        assert "/camera/front/image_raw" not in topics
        assert env.robot_spec.name == "so101"
    finally:
        env.close()
    assert transport.closed


@pytest.mark.integration
def test_ros2_mujoco_adapter_subscribes_and_assembles_commands():
    from physai.robots import create_robot

    transport = RecordingTransport()
    env = create_robot(
        "so101", adapter="ros2_mujoco", transport=transport, render=False
    )
    try:
        assert {
            "/arm_controller/joint_trajectory",
            "/gripper_controller/gripper_cmd",
        } <= set(transport.subscriptions)
        trajectory = SimpleNamespace(
            joint_names=(
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
            ),
            points=[SimpleNamespace(positions=[0.1, -0.9, 1.1, 0.7, 0.0])],
        )
        gripper = SimpleNamespace(command=SimpleNamespace(position=0.3, max_effort=2.0))
        transport.subscriptions["/arm_controller/joint_trajectory"](trajectory)
        transport.subscriptions["/gripper_controller/gripper_cmd"](gripper)

        action = env.pending_action()
        assert action is not None
        np.testing.assert_allclose(action.joint_position, [0.1, -0.9, 1.1, 0.7, 0.0])
        assert action.joint_names == trajectory.joint_names
        assert action.gripper.position == 0.3
    finally:
        env.close()


@pytest.mark.integration
def test_ros2_mujoco_teleop_command_moves_so101():
    from physai.robots import create_robot

    transport = RecordingTransport()
    env = create_robot(
        "so101", adapter="ros2_mujoco", transport=transport, render=False
    )
    try:
        observation = env.reset(seed=42)
        initial_position = observation.joint_state.position[:5].copy()
        initial_gripper = observation.joint_state.position[5]
        trajectory = SimpleNamespace(
            joint_names=(
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
            ),
            points=[SimpleNamespace(positions=[0.2, -0.8, 1.0, 0.5, 0.1])],
        )
        gripper = SimpleNamespace(command=SimpleNamespace(position=0.2, max_effort=2.0))
        transport.subscriptions["/arm_controller/joint_trajectory"](trajectory)
        transport.subscriptions["/gripper_controller/gripper_cmd"](gripper)

        for _ in range(30):
            action = env.pending_action()
            assert action is not None
            observation, *_ = env.step(action)

        np.testing.assert_allclose(
            observation.joint_state.position[:5],
            trajectory.points[0].positions,
            atol=0.05,
        )
        assert not np.allclose(observation.joint_state.position[:5], initial_position)
        assert not np.isclose(observation.joint_state.position[5], initial_gripper)
    finally:
        env.close()


@pytest.mark.integration
def test_mujoco_ros_bridge_ticks_the_latest_ros2_command():
    from physai.bridge import MuJoCoROSBridge
    from physai.contracts import Header, JointState, Observation
    from physai.robots import RobotSpec

    class FakeSimulation:
        robot_spec = RobotSpec(
            name="arm",
            kind="fixed_base_manipulator",
            joint_names=("joint",),
            action_joint_names=("joint",),
            metadata={"control_hz": 20.0},
            joint_state_frame="base",
            units={"joint_position": "rad", "joint_velocity": "rad/s"},
        )

        def __init__(self):
            self.steps = 0
            self.observation = Observation(
                joint_state=JointState(
                    name=("joint",),
                    position=np.zeros(1),
                    velocity=np.zeros(1),
                    effort=np.zeros(1),
                    header=Header(stamp=0.0, frame_id="base"),
                )
            )

        def reset(self, seed=None):
            del seed
            return self.observation

        def observe(self):
            return self.observation

        def send_action(self, action):
            self.robot_spec.validate_action(action)

        def step(self, action):
            self.send_action(action)
            self.steps += 1
            return self.observation, 0.0, False, False, {}

        def close(self):
            pass

    transport = RecordingTransport()
    simulation = FakeSimulation()
    bridge = MuJoCoROSBridge(simulation, transport)
    try:
        bridge.reset(seed=3)
        trajectory = SimpleNamespace(
            joint_names=("joint",), points=[SimpleNamespace(positions=[0.25])]
        )
        transport.subscriptions["/arm_controller/joint_trajectory"](trajectory)
        _, _, terminated, truncated, info = bridge.tick()
        assert simulation.steps == 1
        assert not terminated and not truncated
        assert info == {}
    finally:
        bridge.close()


@pytest.mark.integration
def test_rclpy_transport_creates_and_cleans_up_ros_entities():
    from physai.bridge import RclpyTransport

    class Publisher:
        def __init__(self):
            self.messages = []

        def publish(self, message):
            self.messages.append(message)

    class Node:
        def __init__(self):
            self.publishers = []
            self.subscriptions = []
            self.destroyed_publishers = []
            self.destroyed_subscriptions = []

        def create_publisher(self, message_type, topic, depth):
            publisher = Publisher()
            self.publishers.append((message_type, topic, depth, publisher))
            return publisher

        def create_subscription(self, message_type, topic, callback, depth):
            subscription = (message_type, topic, callback, depth)
            self.subscriptions.append(subscription)
            return subscription

        def destroy_publisher(self, publisher):
            self.destroyed_publishers.append(publisher)

        def destroy_subscription(self, subscription):
            self.destroyed_subscriptions.append(subscription)

    node = Node()
    transport = RclpyTransport(node, {"/command": str})
    transport.subscribe("/command", lambda _: None)
    transport.publish("/state", 1)
    transport.publish("/state", 2)
    transport.close()

    assert len(node.publishers) == 1
    assert node.publishers[0][1] == "/state"
    assert node.publishers[0][3].messages == [1, 2]
    assert len(node.subscriptions) == 1
    assert len(node.destroyed_publishers) == 1
    assert len(node.destroyed_subscriptions) == 1


@pytest.mark.integration
def test_ros2_hardware_adapter_uses_shared_transport_boundary():
    from physai.bridge import ROS2HardwareAdapter
    from physai.contracts import JointState, Observation
    from physai.robots import RobotSpec

    class FakeHardware:
        robot_spec = RobotSpec(
            name="hardware",
            kind="fixed_base_manipulator",
            joint_names=("joint",),
            action_joint_names=("joint",),
            units={"joint_position": "rad", "joint_velocity": "rad/s"},
        )

        def __init__(self):
            self.observation = Observation(
                joint_state=JointState(
                    name=("joint",),
                    position=np.zeros(1),
                    velocity=np.zeros(1),
                    effort=np.zeros(1),
                )
            )

        def reset(self, seed=None):
            del seed
            return self.observation

        def observe(self):
            return self.observation

        def send_action(self, action):
            self.robot_spec.validate_action(action)

        def step(self, action):
            self.send_action(action)
            return self.observation, 0.0, False, False, {}

        def close(self):
            pass

    transport = RecordingTransport()
    adapter = ROS2HardwareAdapter(FakeHardware(), transport)
    adapter.reset()
    adapter.close()
    assert any(topic == "/joint_states" for topic, _ in transport.messages)
    assert transport.closed


@pytest.mark.integration
def test_adapter_factories_build_only_what_they_support(monkeypatch):
    from physai.robots import create_robot

    with pytest.raises(ValueError, match="requires a ROS2 transport"):
        create_robot("so101", adapter="ros2_mujoco", render=False)

    from physai.robots import RobotSpec
    from physai.robots.so101 import factory

    class FakeHardware:
        robot_spec = RobotSpec(
            name="so101",
            kind="hardware",
            units={"joint_position": "rad", "joint_velocity": "rad/s"},
        )

        def close(self):
            pass

    def fail_if_constructed(*args, **kwargs):
        raise AssertionError("hardware adapter must not construct MuJoCo")

    monkeypatch.setattr(factory, "SO101Env", fail_if_constructed)
    transport = RecordingTransport()
    adapter = factory.make_so101(
        adapter="ros2_hardware", transport=transport, hardware=FakeHardware()
    )
    adapter.close()
    assert transport.closed

    from physai.robots.turtlebot import factory

    def fail_if_constructed(*args, **kwargs):
        raise AssertionError("unsupported adapter must not construct MuJoCo")

    monkeypatch.setattr(factory, "TurtleBot4Env", fail_if_constructed)
    with pytest.raises(ValueError, match="not supported for turtlebot4"):
        factory.make_turtlebot4(
            adapter="ros2_hardware", transport=RecordingTransport(), hardware=object()
        )


@pytest.mark.integration
def test_ros2_message_codec_converts_observations():
    from physai.bridge import ROS2MessageCodec
    from physai.contracts import Header, ImageFrame, JointState

    class FakeTime:
        sec = 0
        nanosec = 0

    class FakeJointState:
        def __init__(self):
            self.header = SimpleNamespace(stamp=FakeTime(), frame_id="")

    class FakeImage:
        pass

    codec = ROS2MessageCodec(FakeJointState, FakeImage)
    joints = codec.encode_joint_state(
        JointState(
            name=("joint_a",),
            position=[0.1],
            velocity=[0.2],
            effort=[0.3],
            header=Header(stamp=12.25, frame_id="base"),
        )
    )
    image = codec.encode_image(
        ImageFrame(
            data=np.zeros((2, 3, 3), dtype=np.uint8),
            camera_name="front",
            header=Header(stamp=12.25, frame_id="camera_front"),
        )
    )

    assert joints.name == ["joint_a"]
    assert joints.position == [0.1]
    assert joints.header.frame_id == "base"
    assert (joints.header.stamp.sec, joints.header.stamp.nanosec) == (12, 250000000)
    assert (image.height, image.width, image.encoding, image.step) == (2, 3, "rgb8", 9)
    assert len(image.data) == 18
