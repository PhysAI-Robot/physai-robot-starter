import numpy as np
import pytest

from physai.contracts import (
    Action,
    ActionSpec,
    CameraSpec,
    GripperCommand,
    Header,
    ImageFrame,
    JointState,
    Observation,
    ObservationSpec,
    Quaternion,
    TensorSpec,
    Twist,
    Vector3,
)


def test_value_types_convert_and_reject_ambiguous_input():
    names = ("shoulder_pan", "gripper")
    js = JointState(name=names, position=np.arange(2, dtype=float))
    assert (js.get("shoulder_pan"), js.get("gripper")) == (0.0, 1.0)
    with pytest.raises(ValueError, match="same size"):
        JointState(name=("joint",), position=[0.0], velocity=[0.0, 0.0], effort=[0.0])

    # ROS is x,y,z,w; MuJoCo is w,x,y,z. Getting this backwards is the classic
    # silent bug when porting sim code to ROS2.
    wxyz = np.array([0.5, 0.5, 0.5, 0.5])
    q = Quaternion.from_mujoco(wxyz)
    assert q.as_array().tolist() == [0.5, 0.5, 0.5, 0.5]
    np.testing.assert_allclose(q.to_mujoco(), wxyz)
    q2 = Quaternion.from_mujoco([0.0, 1.0, 0.0, 0.0])
    assert q2.w == 0.0 and q2.x == 1.0

    t = Twist(Vector3(1, 2, 3), Vector3(4, 5, 6))
    np.testing.assert_allclose(t.as_array(), [1, 2, 3, 4, 5, 6])
    np.testing.assert_allclose(Twist.from_array(t.as_array()).as_array(), t.as_array())

    for value, expected in ((-1.0, 0.0), (0.0, 0.0), (0.5, 0.5), (2.0, 1.0)):
        assert GripperCommand(position=value).clipped() == expected

    a = Action(joint_position=[0, 1, 2, 3, 4])
    assert isinstance(a.joint_position, np.ndarray)
    assert a.joint_position.shape == (5,)
    assert Action(ee_twist=Twist()).mode == "twist"
    with pytest.raises(ValueError, match="both"):
        _ = Action(joint_position=[0], ee_twist=Twist()).mode


def test_observation_validates_camera_name_frame_and_timestamp():
    observation = Observation(
        joint_state=JointState(
            name=("joint",),
            position=[0.0],
            velocity=[0.0],
            effort=[0.0],
            header=Header(stamp=1.0, frame_id="base"),
        ),
        images={
            "front": ImageFrame(
                data=np.zeros((2, 2, 3), dtype=np.uint8),
                camera_name="front",
                header=Header(stamp=1.0, frame_id="camera_front"),
            )
        },
        sim_time=1.0,
    )
    observation.validate(
        expected_joint_names=("joint",),
        expected_joint_frame="base",
        expected_camera_frames={"front": "camera_front"},
    )

    observation.images["front"].header.frame_id = "wrong_frame"
    with pytest.raises(ValueError, match="expects frame"):
        observation.validate(expected_camera_frames={"front": "camera_front"})


def test_training_specs_validate_canonical_values_and_reject_bad_ones():
    observation_spec = ObservationSpec(
        fields=(
            TensorSpec(
                name="joint_state",
                shape=(2,),
                dtype="float32",
                units="rad",
                minimum=-1.0,
                maximum=1.0,
                normalization={"method": "min_max", "range": [-1.0, 1.0]},
            ),
        ),
        cameras=(
            CameraSpec(name="wrist_rgb", shape=(2, 3, 3), frame_id="camera_wrist"),
        ),
        metadata={"robot": "so101", "task": "sorting"},
    )
    observation_spec.validate(
        {
            "joint_state": np.array([0.0, 0.5], dtype=np.float32),
            "wrist_rgb": np.zeros((2, 3, 3), dtype=np.uint8),
        }
    )
    serialized = observation_spec.to_dict()
    assert serialized["metadata"]["task"] == "sorting"
    assert serialized["cameras"][0]["frame_id"] == "camera_wrist"

    value_spec = TensorSpec(
        name="action", shape=(2,), dtype="float32", minimum=-1.0, maximum=1.0
    )
    action_spec = ActionSpec(
        fields=(value_spec,), metadata={"layout": "so101_joint_position"}
    )
    action_spec.validate({"action": np.zeros(2, dtype=np.float32)})
    assert action_spec.to_dict()["metadata"]["layout"] == "so101_joint_position"

    with pytest.raises(ValueError, match="shape"):
        value_spec.validate(np.zeros(3, dtype=np.float32))
    with pytest.raises(ValueError, match="dtype"):
        value_spec.validate(np.zeros(2, dtype=np.float64))
    with pytest.raises(ValueError, match="maximum"):
        value_spec.validate(np.array([0.0, 2.0], dtype=np.float32))
    with pytest.raises(ValueError, match="unexpected fields"):
        action_spec.validate({"action": np.zeros(2, dtype=np.float32), "extra": 0})
