"""Runtime bridge for driving a MuJoCo robot through ROS2-shaped topics.

The bridge owns the control loop while ``ROS2MuJoCoAdapter`` owns topic
translation. A real ROS2 node can provide ``RclpyTransport`` and a
``ROS2MessageCodec`` without making ``rclpy`` a core dependency.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

from ..control.safety import SafetyController
from ..contracts import Action, Observation
from ..robots.base import RobotPort, RobotSpec
from .adapters import ROS2MuJoCoAdapter, ROS2Transport
from .messages import MessageCodec


class RclpyTransport:
    """Adapt an existing ``rclpy`` node to the transport protocol.

    Message classes are supplied by the caller because ROS2 action and topic
    interfaces can differ between controllers. Importing this class does not
    import ``rclpy``.
    """

    def __init__(
        self,
        node: Any,
        topic_types: Mapping[str, Any],
        *,
        qos_depth: int = 10,
    ) -> None:
        if qos_depth < 1:
            raise ValueError("qos_depth must be positive")
        self._node = node
        self._topic_types = dict(topic_types)
        self._qos_depth = qos_depth
        self._publishers: dict[str, Any] = {}
        self._subscriptions: list[Any] = []

    def publish(self, topic: str, message: Any) -> None:
        publisher = self._publishers.get(topic)
        if publisher is None:
            publisher = self._node.create_publisher(
                type(message), topic, self._qos_depth
            )
            self._publishers[topic] = publisher
        publisher.publish(message)

    def subscribe(self, topic: str, callback: Callable[[Any], None]) -> None:
        try:
            message_type = self._topic_types[topic]
        except KeyError as exc:
            raise ValueError(f"no ROS2 message type configured for {topic!r}") from exc
        subscription = self._node.create_subscription(
            message_type, topic, callback, self._qos_depth
        )
        self._subscriptions.append(subscription)

    def close(self) -> None:
        for publisher in self._publishers.values():
            self._node.destroy_publisher(publisher)
        for subscription in self._subscriptions:
            self._node.destroy_subscription(subscription)
        self._publishers.clear()
        self._subscriptions.clear()


class MuJoCoROSBridge:
    """Run a synchronous MuJoCo robot behind the Phase 1 ROS2 boundary."""

    def __init__(
        self,
        simulation: RobotPort,
        transport: ROS2Transport,
        *,
        codec: MessageCodec | None = None,
        control_hz: float | None = None,
        safety: SafetyController | None = None,
    ) -> None:
        rate = control_hz
        if rate is None:
            rate = float(simulation.robot_spec.metadata.get("control_hz", 25.0))
        if rate <= 0:
            raise ValueError("control_hz must be positive")
        self._adapter = ROS2MuJoCoAdapter(simulation, transport, codec=codec)
        self._control_hz = rate
        self._safety = safety or SafetyController(self.robot_spec)
        self._observation: Observation | None = None

    @property
    def robot_spec(self) -> RobotSpec:
        return self._adapter.robot_spec

    @property
    def observation(self) -> Observation | None:
        return self._observation

    def reset(self, seed: int | None = None) -> Observation:
        self._observation = self._adapter.reset(seed=seed)
        return self._observation

    def pending_action(self) -> Action | None:
        """Return the latest command assembled from subscribed topics."""
        return self._adapter.pending_action()

    def tick(self) -> tuple[Observation, float, bool, bool, dict]:
        """Advance one control period or publish state while awaiting a command."""
        if self._observation is None:
            raise RuntimeError("call reset() before tick()")
        action = self.pending_action()
        if action is None:
            observation = self._adapter.observe()
            self._observation = observation
            return observation, 0.0, False, False, {"waiting_for_action": True}

        action = self._safety.validate(self._observation, action)
        result = self._adapter.step(action)
        self._observation = result[0]
        return result

    def run(
        self,
        *,
        max_ticks: int | None = None,
        stop: Callable[[], bool] | None = None,
        spin_once: Callable[[], None] | None = None,
    ) -> int:
        """Run the bridge loop until stopped or the robot episode ends."""
        if self._observation is None:
            self.reset()
        if max_ticks is not None and max_ticks < 1:
            raise ValueError("max_ticks must be positive or None")

        period = 1.0 / self._control_hz
        next_tick = time.monotonic()
        ticks = 0
        while max_ticks is None or ticks < max_ticks:
            if stop is not None and stop():
                break
            if spin_once is not None:
                spin_once()
            _, _, terminated, truncated, _ = self.tick()
            ticks += 1
            if terminated or truncated:
                break
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()
        return ticks

    def close(self) -> None:
        self._adapter.close()


__all__ = ["MuJoCoROSBridge", "RclpyTransport"]
