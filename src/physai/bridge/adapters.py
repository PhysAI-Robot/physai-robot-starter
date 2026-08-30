"""Transport adapters for simulated and physical robot ports."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from ..contracts import Action, Observation
from ..robots.base import RobotPort, RobotSpec


class ROS2Transport(Protocol):
    """Small transport port implemented by an rclpy node or a test double."""

    def publish(self, topic: str, message: Any) -> None: ...

    def subscribe(self, topic: str, callback: Callable[[Any], None]) -> None: ...

    def close(self) -> None: ...


class ROS2MuJoCoAdapter:
    """Expose a synchronous MuJoCo port through an injected ROS2 transport."""

    def __init__(self, simulation: RobotPort, transport: ROS2Transport) -> None:
        self._simulation = simulation
        self._transport = transport

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
        """Apply an action received from the ROS2 command subscriber."""
        self.send_action(action)

    def publish_observation(self, observation: Observation) -> None:
        """Publish the canonical observation for a ROS2 driver node."""
        self._transport.publish("/joint_states", observation.joint_state)
        for name, frame in observation.images.items():
            self._transport.publish(f"/camera/{name}/image_raw", frame)

    def close(self) -> None:
        self._simulation.close()
        self._transport.close()


class ROS2HardwareAdapter:
    """Expose an injected hardware port through the same ROS2 boundary."""

    def __init__(self, hardware: RobotPort, transport: ROS2Transport) -> None:
        self._hardware = hardware
        self._transport = transport

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

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict]:
        self.robot_spec.validate_action(action)
        result = self._hardware.step(action)
        self.publish_observation(result[0])
        return result

    def publish_observation(self, observation: Observation) -> None:
        self._transport.publish("/joint_states", observation.joint_state)
        for name, frame in observation.images.items():
            self._transport.publish(f"/camera/{name}/image_raw", frame)

    def close(self) -> None:
        self._hardware.close()
        self._transport.close()
