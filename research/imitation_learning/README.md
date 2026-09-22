# Imitation learning

ACT/LeRobot training pipeline and checkpoint-backed inference policies.

- `act_dataset.py` — torch `Dataset` + `DatasetStats` over recorded episodes.
- `vla_adapter.py` — `VLAPolicy` / `LeRobotPolicy`, checkpoint-backed policies
  (the model-free `ReplayPolicy` stays in core: `physai.policy.replay`).
- `train_act.py` — ACT training entrypoint.

`vla_adapter.py` registers the `"lerobot"` policy with `physai.policy.registry`
on import; core never imports this package. Requires the `training`/`vla`
extras (`uv sync --extra training --extra vla`).

## SO-101 collect -> train -> evaluate workflow

After the scripted task is reliable (see
[docs/SO101_RUNBOOK.md](../../docs/SO101_RUNBOOK.md)), collect
demonstrations and fine-tune an ACT policy:

```bash
uv run python scripts/collect_demos.py --episodes 50 --out data/pickplace_v1
uv run python research/imitation_learning/train_act.py --dataset data/pickplace_v1 --steps 4000
uv run python scripts/eval_policy.py --policy lerobot --checkpoint outputs/act_ckpt --camera-size 128
```

`collect_demos.py` drives the scripted expert
(`research/scripted_experts/`) and discards failed episodes by default —
behavior cloning on failures teaches failure. `train_act.py` stores the
checkpoint and metadata under `outputs/act_ckpt` by default. Use the same
square image size during training and evaluation (`--camera-size` /
`--image-size`); a mismatch silently feeds the policy a distorted,
off-distribution image since `LeRobotPolicy._resize()` center-crops before
resizing rather than stretching.

The current prototype supports ACT-shaped data and scripted, replay, and ACT
policy evaluation. It does not yet export the standard `LeRobotDataset`
format or provide a unified training entry point across techniques.
