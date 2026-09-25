"""The model-free half of the VLA policy seam: replay, no checkpoint needed.

`VLAPolicy` is the generic chunked-action adapter shape (observation
packing, action-chunk buffering, unit decoding) that both a real
checkpoint-backed policy and a no-model replay policy share. It stays in
core because `ReplayPolicy` — the integration test for this whole seam —
needs it and core must not import research code.

Checkpoint-backed subclasses (`LeRobotPolicy`) are research content: see
`research/imitation_learning/vla_adapter.py`.

Observation keys follow the LeRobot convention so a checkpoint fine-tuned on a
dataset recorded from a robot contract needs no remapping:

    observation.images.<camera> (H, W, 3) uint8
    observation.state           (N,) float32
    action                      (M,) float32
"""

from __future__ import annotations

from abc import abstractmethod
from collections import deque
from collections.abc import Callable

import numpy as np

from ..contracts import Action, GripperCommand, Observation, PoseStamped
from ..robots.base import RobotPort
from .base import Policy


class VLAPolicy(Policy):
    """Base adapter. Subclass and implement `_infer`.

    `action_horizon` > 1 enables action chunking: the model is queried once
    every `action_horizon` ticks and the chunk is played out open-loop. This is
    how SmolVLA and ACT are normally run, and it is the difference between
    ~5 Hz of model calls and 25 Hz of control.
    """

    name = "vla"

    def __init__(
        self,
        robot: RobotPort,
        action_horizon: int = 1,
        instruction: str = "",
        image_keys: tuple[str, ...] | None = None,
        action_decoder: Callable[[np.ndarray], Action] | None = None,
    ) -> None:
        self.robot = robot
        self.env = robot
        self.action_horizon = max(1, action_horizon)
        self.instruction = instruction
        contract = getattr(robot, "training_contract", None)
        if image_keys is None:
            self.image_keys = (
                tuple(camera.name for camera in contract.observation_spec.cameras)
                if contract is not None
                else ()
            )
        else:
            self.image_keys = image_keys
        self.action_decoder = action_decoder
        self._chunk: deque[np.ndarray] = deque()

    # -- to implement --------------------------------------------------
    @abstractmethod
    def _infer(self, batch: dict) -> np.ndarray:
        """Return an action chunk in the robot's declared action layout."""

    # -- plumbing ------------------------------------------------------
    def reset(
        self,
        observation: Observation,
        goal: PoseStamped | None = None,
        instruction: str | None = None,
    ) -> None:
        self._chunk.clear()
        if instruction is not None:
            self.instruction = instruction

    def build_batch(self, observation: Observation) -> dict:
        batch: dict = {}
        for key in self.image_keys:
            frame = observation.images.get(key)
            if frame is None:
                raise KeyError(
                    f"camera {key!r} missing from observation; env cameras are "
                    f"{tuple(observation.images)}"
                )
            batch[f"observation.images.{key}"] = frame.data
        batch["observation.state"] = observation.joint_state.position.astype(np.float32)
        batch["task"] = self.instruction
        return batch

    def act(self, observation: Observation) -> Action:
        if not self._chunk:
            chunk = np.asarray(
                self._infer(self.build_batch(observation)), dtype=np.float64
            )
            if chunk.ndim == 1:
                chunk = chunk[None, :]
            expected_width = len(self._model_action_names())
            if chunk.shape[1] != expected_width:
                raise ValueError(
                    f"expected (T, {expected_width}) actions, got {chunk.shape}"
                )
            self._chunk.extend(chunk[: self.action_horizon])

        return self._decode_action(self._chunk.popleft())

    def _decode_action(self, values: np.ndarray) -> Action:
        if self.action_decoder is not None:
            return self.action_decoder(values)
        contract = getattr(self.robot, "training_contract", None)
        if contract is not None and contract.action_decoder is not None:
            return contract.action_decoder(values)
        action_names = self._model_action_names()
        arm_size = len(self.robot.robot_spec.action_joint_names)
        if values.size != len(action_names):
            raise ValueError(
                f"expected {len(action_names)} action values, got {values.size}"
            )
        if values.size == arm_size:
            return Action(joint_position=values, joint_names=action_names)
        if (
            values.size != arm_size + 1
            or "gripper" not in self.robot.robot_spec.capabilities
        ):
            raise ValueError("provide action_decoder for this robot's action layout")
        gripper_to_normalized = getattr(self.robot, "joint_to_gripper", None)
        if gripper_to_normalized is None:
            raise ValueError(
                "robot has no joint_to_gripper mapping; provide action_decoder"
            )
        return Action(
            joint_position=values[:arm_size],
            gripper=GripperCommand(position=gripper_to_normalized(values[arm_size])),
            joint_names=self.robot.robot_spec.action_joint_names,
        )

    def _model_action_names(self) -> tuple[str, ...]:
        contract = getattr(self.robot, "training_contract", None)
        if contract is not None:
            names = contract.action_spec.metadata.get("names")
            if names is not None:
                return tuple(names)
            joint_names = contract.action_spec.metadata.get("joint_names")
            if joint_names is not None:
                return tuple(joint_names)
        names = self.robot.robot_spec.action_joint_names
        if "gripper" in self.robot.robot_spec.capabilities:
            return (*names, "gripper")
        return names


class ReplayPolicy(VLAPolicy):
    """Replays a recorded episode. The integration test for this whole seam.

    If a recorded demo replays successfully, your action space, units, and
    control rate line up — which is exactly what silently breaks first when
    wiring a real VLA.
    """

    name = "replay"

    def __init__(self, env, actions: np.ndarray, **kw) -> None:
        # Replay needs no pixels — and demanding them would make this unusable
        # for datasets recorded with --no-images.
        kw.setdefault("image_keys", ())
        super().__init__(env, **kw)
        self.actions = np.asarray(actions, dtype=np.float64)
        self._i = 0

    def reset(self, observation, goal=None, instruction=None) -> None:
        super().reset(observation, goal, instruction)
        self._i = 0

    @property
    def done(self) -> bool:
        return self._i >= len(self.actions)

    def _infer(self, batch: dict) -> np.ndarray:
        i = min(self._i, len(self.actions) - 1)
        self._i += 1
        return self.actions[i][None, :]


__all__ = ["ReplayPolicy", "VLAPolicy"]
