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


class ROS2MuJoCoAdapter:
    """Expose a synchronous MuJoCo port through an injected ROS2 transport."""

    def __init__(
        self,
        simulation: RobotPort,
        transport: ROS2Transport,
        codec: MessageCodec | None = None,
    ) -> None:
        self._simulation = simulation
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
        return self._simulation.robot_spec

    def reset(self, seed: int | None = None) -> Observation:
        observation = self._simulation.reset(seed=seed)
        self.publish_observation(observation)
        return observation

    def observe(self) -> Observation:
        observation = self._simulation.observe()
        self.publish_observation(observation)
        return observation

    def send_action(self, action: Action) -> None:
        self.robot_spec.validate_action(action)
        self._simulation.send_action(action)

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict]:
        self.robot_spec.validate_action(action)
        result = self._simulation.step(action)
        self.publish_observation(result[0])
        return result

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

    def publish_observation(self, observation: Observation) -> None:
        """Publish the canonical observation for a ROS2 driver node."""
        self._transport.publish(
            "/joint_states", self._codec.encode_joint_state(observation.joint_state)
        )
        for name, frame in observation.images.items():
            self._transport.publish(
                f"/camera/{name}/image_raw", self._codec.encode_image(frame)
            )

    def close(self) -> None:
        self._simulation.close()
        self._transport.close()


class ROS2HardwareAdapter:
    """Expose an injected hardware port through the same ROS2 boundary."""

    def __init__(
        self,
        hardware: RobotPort,
        transport: ROS2Transport,
        codec: MessageCodec | None = None,
    ) -> None:
        self._hardware = hardware
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
        return self._hardware.robot_spec

    def reset(self, seed: int | None = None) -> Observation:
        observation = self._hardware.reset(seed=seed)
        self.publish_observation(observation)
        return observation

    def observe(self) -> Observation:
        observation = self._hardware.observe()
        self.publish_observation(observation)
        return observation

    def send_action(self, action: Action) -> None:
        self.robot_spec.validate_action(action)
        self._hardware.send_action(action)

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
        result = self._hardware.step(action)
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
        self._hardware.close()
        self._transport.close()
