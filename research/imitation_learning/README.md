# Imitation learning

ACT/LeRobot training pipeline and checkpoint-backed inference policies.

- `act_dataset.py` — torch `Dataset` + `DatasetStats` over recorded episodes.
- `vla_adapter.py` — `VLAPolicy` / `LeRobotPolicy`, checkpoint-backed policies
  (the model-free `ReplayPolicy` stays in core: `physai.policy.replay`).
- `train_act.py` — ACT training entrypoint. Run with
  `uv run python research/imitation_learning/train_act.py --dataset <path>`.

Registers its policies with `physai.policy.registry` on import; core never
imports this package. Requires the `training`/`vla` extras
(`uv sync --extra training --extra vla`).
