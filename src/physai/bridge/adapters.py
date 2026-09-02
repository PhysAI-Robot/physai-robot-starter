"""Transport adapters for simulated and physical robot ports."""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import Any, Protocol

from ..contracts import Action, GripperCommand, Observation
from ..robots.base import RobotPort, RobotSpec
from .messages import ContractMessageCodec, MessageCodec


class ROS2Transport(Protocol):
    """Small transport port implemented by an rclpy node or a test double."""

    def publish(self, topic: str, message: Any) -> None: ...

    def subscribe(self, topic: str, callback: Callable[[Any], None]) -> None: ...

    def close(self) -> None: ...


class _ROS2RobotAdapter:
    """Share ROS2 transport behavior across simulation and hardware ports."""

    def __init__(
        self,
        robot: RobotPort,
        transport: ROS2Transport,
        codec: MessageCodec | None = None,
    ) -> None:
        self._robot = robot
        self._transport = transport
        self._codec = codec or ContractMessageCodec()
        self._command_lock = Lock()
        self._pending_joint: Action | None = None
        self._pending_gripper: GripperCommand | None = None
        transport.subscribe(
            "/arm_controller/joint_trajectory", self._receive_joint_trajectory
        )
        transport.subscribe(
            "/gripper_controller/gripper_cmd", self._receive_gripper_command
        )

    @property
    def robot_spec(self) -> RobotSpec:
        return self._robot.robot_spec

    def reset(self, seed: int | None = None) -> Observation:
        observation = self._robot.reset(seed=seed)
        self.robot_spec.validate_observation(observation)
        self.publish_observation(observation)
        return observation

    def observe(self) -> Observation:
        observation = self._robot.observe()
        self.robot_spec.validate_observation(observation)
        self.publish_observation(observation)
        return observation

    def send_action(self, action: Action) -> None:
        self.robot_spec.validate_action(action)
        self._robot.send_action(action)

    def receive_action(self, action: Action) -> None:
        """Queue a complete internal action received from a ROS2 node."""
        self.robot_spec.validate_action(action)
        with self._command_lock:
            self._pending_joint = action

    def pending_action(self) -> Action | None:
        """Return the latest complete command assembled from subscribed topics."""
        with self._command_lock:
            if self._pending_joint is None:
                return None
            action = self._pending_joint
            if self._pending_gripper is not None:
                action = Action(
                    joint_position=action.joint_position,
                    joint_names=action.joint_names,
                    stamp=action.stamp,
                    gripper=self._pending_gripper,
                )
            return action

    def _receive_joint_trajectory(self, message: Any) -> None:
        self.receive_action(self._codec.decode_joint_trajectory(message))

    def _receive_gripper_command(self, message: Any) -> None:
        gripper = self._codec.decode_gripper_command(message)
        with self._command_lock:
            self._pending_gripper = gripper

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict]:
        self.robot_spec.validate_action(action)
        result = self._robot.step(action)
        self.robot_spec.validate_observation(result[0])
        self.publish_observation(result[0])
        return result

    def publish_observation(self, observation: Observation) -> None:
        self._transport.publish(
            "/joint_states", self._codec.encode_joint_state(observation.joint_state)
        )
        for name, frame in observation.images.items():
            self._transport.publish(
                f"/camera/{name}/image_raw", self._codec.encode_image(frame)
            )

    def close(self) -> None:
        self._robot.close()
        self._transport.close()


class ROS2MuJoCoAdapter(_ROS2RobotAdapter):
    """Expose a synchronous MuJoCo port through an injected ROS2 transport."""


class ROS2HardwareAdapter(_ROS2RobotAdapter):
    """Expose an injected hardware port through the same ROS2 boundary."""
