"""Small protocol fakes shared by unit and integration tests."""

from __future__ import annotations

import numpy as np

from physai.contracts import JointState, Observation
from physai.robots import RobotSpec


class RecordingTransport:
    def __init__(self) -> None:
        self.messages = []
        self.subscriptions = {}
        self.closed = False

    def publish(self, topic, message) -> None:
        self.messages.append((topic, message))

    def subscribe(self, topic, callback) -> None:
        self.subscriptions[topic] = callback

    def close(self) -> None:
        self.closed = True


class FakeRobotPort:
    def __init__(self, spec: RobotSpec, *, validate_actions: bool = True) -> None:
        self.robot_spec = spec
        self.validate_actions = validate_actions
        self.closed = False
        self.observation = Observation(
            joint_state=JointState(
                name=spec.joint_names,
                position=np.zeros(len(spec.joint_names)),
                velocity=np.zeros(len(spec.joint_names)),
                effort=np.zeros(len(spec.joint_names)),
            )
        )

    def reset(self, seed=None):
        del seed
        return self.observation

    def observe(self):
        return self.observation

    def send_action(self, action) -> None:
        if self.validate_actions:
            self.robot_spec.validate_action(action)

    def step(self, action):
        self.send_action(action)
        return self.observation, 0.0, False, False, {}

    def close(self) -> None:
        self.closed = True


__all__ = ["FakeRobotPort", "RecordingTransport"]
