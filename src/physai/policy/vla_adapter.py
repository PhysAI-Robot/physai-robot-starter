"""Seam for dropping a real VLA checkpoint into the loop.

Nothing here imports torch — Phase 0 runs without it. The point of this file is
that when you `pip install lerobot` and load SmolVLA (or TurboVLA, or an ACT
checkpoint you trained on the demos from `scripts/collect_demos.py`), the only
thing you write is `_infer`. Everything else — observation packing, action
chunk buffering, unit conversion — is already handled and matches the format
the recorder writes.

Observation keys follow the LeRobot convention so a checkpoint fine-tuned on a
LeRobot dataset recorded from this env needs no remapping:

    observation.images.front   (H, W, 3) uint8
    observation.images.wrist   (H, W, 3) uint8
    observation.state          (6,) float32   5 arm joints + gripper, radians
    action                     (6,) float32   same layout, absolute targets
"""

from __future__ import annotations

from abc import abstractmethod
from collections import deque

import numpy as np

from ..contracts import Action, GripperCommand, Observation, PoseStamped
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
        env,
        action_horizon: int = 1,
        instruction: str = "",
        image_keys: tuple[str, ...] = ("front", "wrist"),
    ) -> None:
        self.env = env
        self.action_horizon = max(1, action_horizon)
        self.instruction = instruction
        self.image_keys = image_keys
        self._chunk: deque[np.ndarray] = deque()

    # -- to implement --------------------------------------------------
    @abstractmethod
    def _infer(self, batch: dict) -> np.ndarray:
        """Return (action_horizon, 6) absolute joint targets in radians.

        Column layout: 5 arm joints then the gripper joint (radians, *not*
        normalised) — matching `observation.state`.
        """

    # -- plumbing ------------------------------------------------------
    def reset(self, observation: Observation, goal: PoseStamped | None = None,
              instruction: str | None = None) -> None:
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
            chunk = np.asarray(self._infer(self.build_batch(observation)), dtype=np.float64)
            if chunk.ndim == 1:
                chunk = chunk[None, :]
            if chunk.shape[1] != 6:
                raise ValueError(f"expected (T, 6) actions, got {chunk.shape}")
            self._chunk.extend(chunk[: self.action_horizon])

        a = self._chunk.popleft()
        return Action(
            joint_position=a[:5],
            gripper=GripperCommand(position=self.env.joint_to_gripper(a[5])),
        )


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


class LeRobotPolicy(VLAPolicy):
    """Thin wrapper over a LeRobot `PreTrainedPolicy` (SmolVLA, ACT, pi0, ...).

    Requires `pip install lerobot` and torch. Kept import-light so the rest of
    Phase 0 runs on a machine with neither.

        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        model = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
        policy = LeRobotPolicy(env, model, action_horizon=10,
                               instruction="put the red cube on the green pad")
    """

    name = "lerobot"

    def __init__(self, env, model, device: str = "cpu", **kw) -> None:
        super().__init__(env, **kw)
        self.model = model
        self.device = device

    def _infer(self, batch: dict) -> np.ndarray:
        import torch

        tensors = {}
        for k, v in batch.items():
            if k.startswith("observation.images."):
                img = torch.from_numpy(v).permute(2, 0, 1).float().div(255.0)
                tensors[k] = img.unsqueeze(0).to(self.device)
            elif k == "observation.state":
                tensors[k] = torch.from_numpy(v).float().unsqueeze(0).to(self.device)
            else:
                tensors[k] = [v]

        self.model.eval()
        with torch.inference_mode():
            action = self.model.select_action(tensors)
        return action.squeeze(0).cpu().numpy()
