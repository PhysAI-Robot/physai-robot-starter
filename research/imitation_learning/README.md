# Imitation learning

ACT/LeRobot training pipeline and checkpoint-backed inference policies.

- `act_dataset.py` — torch `Dataset` + `DatasetStats` over recorded episodes.
- `vla_adapter.py` — `LeRobotPolicy`, the checkpoint-backed policy. It extends
  `VLAPolicy`, and the model-free `ReplayPolicy` stays in core with it
  (`physai.policy.replay`).
- `train_act.py` — ACT training entrypoint.

`vla_adapter.py` registers the `"lerobot"` policy with `physai.policy.registry`
on import; core never imports this package. Requires the `training`/`vla`
extras (`uv sync --extra training --extra vla`). `vla` cannot combine with
`isaac` in the same sync — `lerobot==0.6.1` pins `numpy<2.3.0`, incompatible
with isaacsim's `numpy==2.3.1` — so `[tool.uv] conflicts` rejects that
combination instead of `uv lock` failing on the whole project.

## SO-101 collect -> train -> evaluate workflow

After the scripted task is reliable (see
[docs/SO101_RUNBOOK.md](../../docs/SO101_RUNBOOK.md)), collect
demonstrations and fine-tune an ACT policy:

```bash
uv run python scripts/collect_demos.py --episodes 50 --out data/pickplace_v1
uv run python research/imitation_learning/train_act.py --dataset data/pickplace_v1 --steps 4000
uv run python scripts/eval_policy.py --policy lerobot --checkpoint outputs/act_ckpt
```

`collect_demos.py` drives the scripted expert
(`research/scripted_experts/`) and discards failed episodes by default —
behavior cloning on failures teaches failure. `train_act.py` stores the
checkpoint and metadata under `outputs/act_ckpt` by default. The camera
resolution is fixed (`physai.contracts.CAMERA_SIZE`, 320 x 240), so data
collection and evaluation always match; `--image-size` is only the policy's
square input size. Both `act_dataset` and `LeRobotPolicy._resize()`
center-crop the camera image to a square before resizing, so keep them
consistent.

The current prototype supports ACT-shaped data and scripted, replay, and ACT
policy evaluation. It does not yet export the standard `LeRobotDataset`
format or provide a unified training entry point across techniques.
