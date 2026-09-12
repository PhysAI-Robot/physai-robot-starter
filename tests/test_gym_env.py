import numpy as np
import pytest

from physai.contracts import (
    Action,
    ActionSpec,
    Header,
    JointState,
    Observation,
    ObservationSpec,
    TensorSpec,
)
from physai.data.gym_env import GymnasiumAdapter
from physai.robots.base import RobotSpec


class FakeBackend:
    robot_spec = RobotSpec(
        name="fake",
        kind="fixed_base_manipulator",
        joint_names=("joint",),
        action_joint_names=("joint",),
        capabilities=("joint_position",),
        joint_limits={"joint": (-1.0, 1.0)},
        max_joint_delta={"joint": 0.5},
        joint_state_frame="base",
    )

    def __init__(self):
        self.value = 0.0
        self.closed = False

    def _observation(self) -> Observation:
        return Observation(
            joint_state=JointState(
                name=("joint",),
                position=[self.value],
                velocity=[0.0],
                effort=[0.0],
                header=Header(frame_id="base"),
            )
        )

    def reset(self, seed=None):
        self.value = 0.0
        return self._observation()

    def step(self, action):
        self.value = float(action.joint_position[0])
        return self._observation(), 1.5, False, False, {"backend": True}

    def close(self):
        self.closed = True


def make_env(*, max_delta=0.5):
    backend = FakeBackend()
    backend.robot_spec.max_joint_delta["joint"] = max_delta
    env = GymnasiumAdapter(
        backend,
        observation_spec=ObservationSpec(
            fields=(TensorSpec("state", (1,), "float32", minimum=-1.0, maximum=1.0),)
        ),
        action_spec=ActionSpec(
            fields=(TensorSpec("action", (1,), "float32", minimum=-1.0, maximum=1.0),)
        ),
        observation_encoder=lambda observation: {
            "state": np.asarray(observation.joint_state.position, dtype=np.float32)
        },
        action_decoder=lambda action: Action(
            joint_position=np.asarray(action["action"], dtype=np.float64)
        ),
    )
    return env, backend


def test_gymnasium_adapter_reset_step_and_close():
    env, backend = make_env(max_delta=2.0)

    observation, reset_info = env.reset(seed=7)
    assert observation["state"].dtype == np.float32
    assert reset_info["seed"] == 7
    assert env.action_space.contains({"action": np.array([0.25], dtype=np.float32)})

    next_observation, reward, terminated, truncated, info = env.step(
        {"action": np.array([0.25], dtype=np.float32)}
    )
    assert next_observation["state"].tolist() == [0.25]
    assert reward == 1.5
    assert not terminated and not truncated
    assert info == {"backend": True}

    env.close()
    assert backend.closed


def test_gymnasium_adapter_runs_safety_before_backend():
    env, backend = make_env(max_delta=0.1)
    env.reset()

    with pytest.raises(ValueError, match="exceeds max step"):
        env.step({"action": np.array([0.5], dtype=np.float32)})
    assert backend.value == 0.0
