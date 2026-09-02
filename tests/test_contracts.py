import numpy as np
import pytest

from physai.contracts import (
    ALL_JOINT_NAMES,
    Action,
    GripperCommand,
    JointState,
    Quaternion,
    Twist,
    Vector3,
)


def test_joint_state_lookup_by_name():
    js = JointState(position=np.arange(6, dtype=float))
    assert js.get("shoulder_pan") == 0.0
    assert js.get("gripper") == 5.0
    assert js.name == ALL_JOINT_NAMES


def test_quaternion_ros_and_mujoco_orders_round_trip():
    # ROS is x,y,z,w; MuJoCo is w,x,y,z. Getting this backwards is the classic
    # silent bug when porting sim code to ROS2.
    wxyz = np.array([0.5, 0.5, 0.5, 0.5])
    q = Quaternion.from_mujoco(wxyz)
    assert q.as_array().tolist() == [0.5, 0.5, 0.5, 0.5]
    np.testing.assert_allclose(q.to_mujoco(), wxyz)

    q2 = Quaternion.from_mujoco([0.0, 1.0, 0.0, 0.0])
    assert q2.w == 0.0 and q2.x == 1.0


def test_twist_array_round_trip():
    t = Twist(Vector3(1, 2, 3), Vector3(4, 5, 6))
    np.testing.assert_allclose(t.as_array(), [1, 2, 3, 4, 5, 6])
    np.testing.assert_allclose(Twist.from_array(t.as_array()).as_array(), t.as_array())


@pytest.mark.parametrize("value,expected", [(-1.0, 0.0), (0.0, 0.0), (0.5, 0.5), (2.0, 1.0)])
def test_gripper_command_is_clipped_to_unit_range(value, expected):
    assert GripperCommand(position=value).clipped() == expected


def test_action_coerces_joint_position_to_array():
    a = Action(joint_position=[0, 1, 2, 3, 4])
    assert isinstance(a.joint_position, np.ndarray)
    assert a.joint_position.shape == (5,)


def test_action_reports_generic_mode_and_rejects_ambiguous_commands():
    from physai.contracts import Action, Twist

    assert Action(ee_twist=Twist()).mode == "twist"
    with pytest.raises(ValueError, match="both"):
        Action(joint_position=[0], ee_twist=Twist()).mode


def test_joint_state_rejects_mismatched_arrays():
    with pytest.raises(ValueError, match="same size"):
        JointState(
            name=("joint",),
            position=[0.0],
            velocity=[0.0, 0.0],
            effort=[0.0],
        )


def test_observation_validates_camera_name_frame_and_timestamp():
    from physai.contracts import Header, ImageFrame, Observation

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


def test_model_store_requires_a_local_model(tmp_path, monkeypatch):
    from physai import model_store

    monkeypatch.setattr(model_store, "MODEL_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="download_models.py"):
        model_store.resolve_local_model("org/missing-model")


def test_model_download_presets_include_vla_models():
    from scripts.download_models import MODEL_REPOS

    assert MODEL_REPOS["smolvla"] == "lerobot/smolvla_base"
    assert MODEL_REPOS["turbovla"] == "H-EmbodVis/TurboVLA"
