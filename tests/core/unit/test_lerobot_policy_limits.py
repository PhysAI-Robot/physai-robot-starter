from types import SimpleNamespace

import numpy as np

from research.imitation_learning.vla_adapter import LeRobotPolicy


def _policy() -> LeRobotPolicy:
    robot_spec = SimpleNamespace(
        action_joint_names=("shoulder_pan", "wrist_flex"),
        capabilities=("joint_position", "gripper"),
        joint_limits={"shoulder_pan": (-1.0, 1.0), "wrist_flex": (-1.5, 1.5)},
    )
    robot = SimpleNamespace(robot_spec=robot_spec)
    return LeRobotPolicy(robot, policy=None, preprocessor=None, postprocessor=None)


def test_model_output_is_clipped_to_joint_limits():
    clipped = _policy()._clip_to_joint_limits(np.array([-1.2, 1.5004, 0.7]))

    # The gripper has no joint limit, so its value passes through unchanged.
    np.testing.assert_allclose(clipped, [-1.0, 1.5, 0.7])


def test_in_range_output_is_unchanged_and_the_input_is_not_mutated():
    values = np.array([[0.2, -0.3, 0.5], [0.9, 1.4, 0.0]])
    clipped = _policy()._clip_to_joint_limits(values)

    np.testing.assert_array_equal(clipped, values)
    assert clipped is not values
