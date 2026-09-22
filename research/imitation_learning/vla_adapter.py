"""Seam for dropping a real VLA checkpoint into the loop.

Nothing here imports torch at module scope. When you `uv sync --extra vla`
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


def make_lerobot(*, env, checkpoint, **kwargs: Any) -> LeRobotPolicy:
    """Build the checkpoint-backed ACT/LeRobot policy."""
    return LeRobotPolicy.from_checkpoint(env, checkpoint, **kwargs)


register_policy("lerobot", make_lerobot)


__all__ = ["LeRobotPolicy", "make_lerobot"]
