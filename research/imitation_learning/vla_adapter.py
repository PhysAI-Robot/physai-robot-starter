"""Seam for dropping a real VLA checkpoint into the loop.

Nothing here imports torch at module scope. When you `uv sync --extra training`
and load an ACT checkpoint trained on the demos from
`scripts/collect_demos.py`, the only thing you write is `_infer`. Everything
else — observation packing, action chunk buffering, unit conversion — comes
from `physai.policy.replay.VLAPolicy` and matches the format the recorder
writes.

Research module: registers the "lerobot" policy with
``physai.policy.registry`` on import. Core never imports this module
directly (see research/README.md); scripts that support ``--policy
lerobot`` (``run_sim.py``, ``eval_policy.py``) import it explicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from physai.contracts import Observation, PoseStamped
from physai.data.metadata import CheckpointMetadata, validate_checkpoint_compatibility
from physai.policy.registry import register_policy
from physai.policy.replay import VLAPolicy


class LeRobotPolicy(VLAPolicy):
    """Wraps a LeRobot `PreTrainedPolicy` plus its pre/post-processing pipeline.

    `action_horizon` defaults to 1 deliberately: policies like ACT already
    manage their own action-chunk queue inside `select_action` (it only
    re-invokes the model when its internal queue empties). Layering this
    class's own chunk buffer on top at horizon 1 makes it a no-op passthrough,
    so the two don't fight over how many steps to play open-loop.

    Build with `LeRobotPolicy.from_checkpoint(...)` — see act_dataset.py /
    train_act.py for how a checkpoint is produced.
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

        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(
                f"no ACT checkpoint at {checkpoint_dir}; "
                "train one with research/imitation_learning/train_act.py"
            )
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        policy = ACTPolicy.from_pretrained(checkpoint_dir).to(device)
        policy.eval()
        _set_execution(
            policy,
            n_action_steps=kw.pop("n_action_steps", None),
            temporal_ensemble_coeff=kw.pop("temporal_ensemble_coeff", None),
        )

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
        return self._clip_to_joint_limits(action.squeeze(0).detach().cpu().numpy())

    def _clip_to_joint_limits(self, values: np.ndarray) -> np.ndarray:
        """Clip the model's output to the robot's joint limits.

        ACT regresses joint targets, so near a limit it can overshoot by a few
        milliradians (a `wrist_flex` of 1.659 against a 1.658 limit ended
        episodes in the first 30 steps). The safety gate must keep refusing
        out-of-range commands; staying in range is the policy's job.
        """
        limits = self.robot.robot_spec.joint_limits
        clipped = np.array(values, dtype=np.float64, copy=True)
        for column, name in enumerate(self._model_action_names()):
            if name in limits and column < clipped.shape[-1]:
                clipped[..., column] = np.clip(clipped[..., column], *limits[name])
        return clipped


def _set_execution(
    policy, *, n_action_steps: int | None, temporal_ensemble_coeff: float | None
) -> None:
    """How many of each predicted chunk to run before predicting again.

    A checkpoint is trained for a chunk of `chunk_size` actions and by default
    plays all of them open-loop. Running fewer (`n_action_steps`), or blending
    overlapping chunks every step (`temporal_ensemble_coeff`, 0.01 in the ACT
    paper), reacts to what the cameras see sooner, with no retraining.
    """
    config = policy.config
    if temporal_ensemble_coeff is not None:
        if n_action_steps not in (None, 1):
            raise ValueError(
                "temporal ensembling predicts every step: n_action_steps=1"
            )
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

        config.temporal_ensemble_coeff = float(temporal_ensemble_coeff)
        config.n_action_steps = 1
        policy.temporal_ensembler = ACTTemporalEnsembler(
            config.temporal_ensemble_coeff, config.chunk_size
        )
    elif n_action_steps is not None:
        if not 1 <= n_action_steps <= config.chunk_size:
            raise ValueError(
                f"n_action_steps must be in 1..{config.chunk_size}, got {n_action_steps}"
            )
        config.n_action_steps = int(n_action_steps)


def make_lerobot(*, env, checkpoint, **kwargs: Any) -> LeRobotPolicy:
    """Build the checkpoint-backed ACT/LeRobot policy."""
    return LeRobotPolicy.from_checkpoint(env, checkpoint, **kwargs)


register_policy("lerobot", make_lerobot)


__all__ = ["LeRobotPolicy", "make_lerobot"]
