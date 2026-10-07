import numpy as np

from physai.bridge import (
    CartesianTargetRequest,
    CartesianTargetService,
)
from physai.contracts import Header, JointState, Pose, PoseStamped, Quaternion, Vector3
from physai.robots import RobotSpec


class FakeIKResult:
    qpos = np.zeros(2)
    position_error = 1e-4
    orientation_error = 2e-4
    iterations = 4
    converged = True


class FakeKinematics:
    def ik(self, target_pos, q_init=None, target_quat_wxyz=None):
        assert target_pos.shape == (3,)
        assert q_init.shape == (2,)
        assert target_quat_wxyz.shape == (4,)
        return FakeIKResult()


def _request(frame_id="base"):
    return CartesianTargetRequest(
        PoseStamped(
            pose=Pose(
                position=Vector3(0.2, 0.1, 0.2),
                orientation=Quaternion(),
            ),
            header=Header(stamp=1.0, frame_id=frame_id),
        )
    )


def _service():
    spec = RobotSpec(
        name="test_arm",
        kind="fixed_base_manipulator",
        action_joint_names=("joint_a", "joint_b"),
        units={"joint_position": "rad", "joint_velocity": "rad/s"},
    )
    return CartesianTargetService(FakeKinematics(), spec)


def test_cartesian_service_returns_safe_joint_action():
    result = _service().handle(
        _request(),
        JointState(
            name=("joint_a", "joint_b"),
            position=np.zeros(2),
            velocity=np.zeros(2),
            effort=np.zeros(2),
        ),
    )

    assert result.accepted
    assert result.reason == "accepted"
    assert result.action is not None
    np.testing.assert_allclose(result.action.joint_position, [0.0, 0.0])


def test_cartesian_service_rejects_wrong_frame():
    result = _service().handle(
        _request("map"),
        JointState(
            name=("joint_a", "joint_b"),
            position=np.zeros(2),
            velocity=np.zeros(2),
            effort=np.zeros(2),
        ),
    )

    assert not result.accepted
    assert "frame" in result.reason
