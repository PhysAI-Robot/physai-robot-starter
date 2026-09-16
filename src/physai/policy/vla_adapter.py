"""Seam for dropping a real VLA checkpoint into the loop.

Nothing here imports torch. When you `uv sync --extra vla` and load SmolVLA
(or TurboVLA, or an ACT
checkpoint you trained on the demos from `scripts/collect_demos.py`), the only
thing you write is `_infer`. Everything else — observation packing, action
chunk buffering, unit conversion — is already handled and matches the format
the recorder writes.

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
from ..data.metadata import CheckpointMetadata, validate_checkpoint_compatibility
from ..model_store import resolve_local_model
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


class LeRobotPolicy(VLAPolicy):
    """Wraps a LeRobot `PreTrainedPolicy` plus its pre/post-processing pipeline.

    `action_horizon` defaults to 1 deliberately: policies like ACT already
    manage their own action-chunk queue inside `select_action` (it only
    re-invokes the model when its internal queue empties). Layering this
    class's own chunk buffer on top at horizon 1 makes it a no-op passthrough,
    so the two don't fight over how many steps to play open-loop.

    Build with `LeRobotPolicy.from_checkpoint(...)` — see act_dataset.py /
    scripts/train_act.py for how a checkpoint is produced.
    """

    name = "lerobot"

    def __init__(
        self,
        env,
        policy,
        preprocessor,
        postprocessor,
        image_size: int | None = None,
        **kw,
    ) -> None:
        kw.setdefault("action_horizon", 1)
        super().__init__(env, **kw)
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.image_size = image_size

    @classmethod
    def from_checkpoint(
        cls, env, checkpoint_dir, device: str | None = None, **kw
    ) -> LeRobotPolicy:
        import json

        import torch
        from lerobot.policies.act import ACTPolicy, make_act_pre_post_processors

        checkpoint_dir = resolve_local_model(str(checkpoint_dir))
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        policy = ACTPolicy.from_pretrained(checkpoint_dir).to(device)
        policy.eval()

        stats = json.loads(
            (checkpoint_dir / "dataset_stats.json").read_text(encoding="utf-8")
        )
        preprocessor, postprocessor = make_act_pre_post_processors(
            policy.config, dataset_stats=stats
        )

        # Not config.json — ACTPolicy.save_pretrained() owns that filename
        # (it's the full ACTConfig dump). Our own metadata lives alongside it
        # under a name that can't collide. See train_act.py for why this
        # matters: this used to silently read None here.
        image_size = None
        meta_path = checkpoint_dir / "training_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            image_size = meta.get("image_size")
            kw.setdefault("instruction", meta.get("task", ""))

        checkpoint_meta_path = checkpoint_dir / "checkpoint_meta.json"
        if checkpoint_meta_path.exists():
            checkpoint_meta = CheckpointMetadata.from_dict(
                json.loads(checkpoint_meta_path.read_text(encoding="utf-8"))
            )
            contract = getattr(env, "training_contract", None)
            if contract is not None:
                expected_observation_schema = dict(contract.observation_schema)
                expected_action_schema = dict(contract.action_schema)
            else:
                expected_observation_schema = {
                    "observation.state": {"shape": [len(env.robot_spec.joint_names)]}
                }
                expected_action_schema = {
                    "schema": env.robot_spec.metadata.get(
                        "action_schema", "generic.v1"
                    ),
                    "names": list(
                        (*env.robot_spec.action_joint_names,)
                        + (
                            ("gripper",)
                            if "gripper" in env.robot_spec.capabilities
                            else ()
                        )
                    ),
                }
            validate_checkpoint_compatibility(
                checkpoint_meta,
                {
                    "schema_version": "physai.checkpoint.v1",
                    "robot": env.robot_spec.name,
                    **(
                        {"task": env.task.name}
                        if getattr(env, "task", None) is not None
                        else {}
                    ),
                    "observation_schema": expected_observation_schema,
                    "action_schema": expected_action_schema,
                },
            )

        return cls(
            env, policy, preprocessor, postprocessor, image_size=image_size, **kw
        )

    def reset(
        self,
        observation: Observation,
        goal: PoseStamped | None = None,
        instruction: str | None = None,
    ) -> None:
        super().reset(observation, goal, instruction)
        self.policy.reset()

    def _resize(self, arr: np.ndarray):
        """Match training preprocessing: centre-crop to square, then resize.

        training_data collection always renders square frames, so a naive
        stretch-to-square resize of a *non*-square camera render (e.g.
        run_sim.py's default 640x480) silently feeds the policy a distorted,
        off-distribution image — verified: it measurably hurts success rate
        even though every shape lines up and nothing errors. Center-cropping
        first makes any camera aspect ratio degrade gracefully instead.
        """
        import torch

        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        if t.shape[-2] != t.shape[-1]:
            h, w = t.shape[-2], t.shape[-1]
            side = min(h, w)
            top, left = (h - side) // 2, (w - side) // 2
            t = t[:, top : top + side, left : left + side]
        if self.image_size and t.shape[-1] != self.image_size:
            t = torch.nn.functional.interpolate(
                t.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return t

    def _infer(self, batch: dict) -> np.ndarray:
        import torch

        sample = {}
        for k, v in batch.items():
            if k.startswith("observation.images."):
                sample[k] = self._resize(v)
            elif k == "observation.state":
                sample[k] = torch.from_numpy(v).float()
            else:
                sample[k] = v

        processed = self.preprocessor(sample)
        action = self.policy.select_action(processed)
        action = self.postprocessor(action)
        return action.squeeze(0).detach().cpu().numpy()
