# Imitation learning

ACT/LeRobot training pipeline and checkpoint-backed inference policies.

- `act_dataset.py`: torch `Dataset` and `DatasetStats` over recorded episodes.
- `vla_adapter.py`: `LeRobotPolicy`, the checkpoint-backed policy. It extends
  `VLAPolicy`; the model-free `ReplayPolicy` stays in core with it
  (`physai.policy.replay`).
- `train_act.py`: ACT training entrypoint.

Measured results: [FINDINGS](FINDINGS.md).

`vla_adapter.py` registers the `"lerobot"` policy with `physai.policy.registry` on
import; core never imports this package. It needs the `training` extra
(`uv sync --extra training`), which combines with `isaac` in one environment, so the same
checkpoint runs on Isaac Sim (`eval_policy.py --sim isaac --policy lerobot`); the
`numpy` and `packaging` overrides that make this possible are in
[DECISIONS.md D](../../docs/DECISIONS.md#d-isaac-sim-is-an-optional-peer-engine).

## SO-101 collect, train, evaluate

Once the scripted task is reliable ([robot runbook](../../docs/ROBOT_RUNBOOKS.md#so-101)),
collect demonstrations and train an ACT policy:

```bash
uv run python scripts/collect_demos.py --episodes 50 --dataset data/pickplace_v1
uv run python research/imitation_learning/train_act.py --dataset data/pickplace_v1 --steps 4000
uv run python scripts/eval_policy.py --policy lerobot --checkpoint outputs/act_ckpt
```

`collect_demos.py` drives the scripted expert (`research/scripted_experts/`) and discards
failed episodes by default, since behavior cloning on failures teaches failure
(`--keep-failures` keeps them). `train_act.py` writes the checkpoint and metadata to
`outputs/act_ckpt`. Collection and evaluation must use the same camera resolution
(default 320 x 240; `--camera-res` on both). `--image-size` is only the policy's square
input size, and `act_dataset` and `LeRobotPolicy._resize()` both center-crop to a square
before resizing, so keep them consistent.

For the randomized task of study 1 (the configuration in
[FINDINGS](FINDINGS.md#randomized-pick-and-place-study-1-m3)):

```bash
uv run python scripts/collect_demos.py --manifest configs/manifests/so101_randomized_pick_place.yaml --episodes 200 --seed 0 --dataset data/randomized_v1
uv run python research/imitation_learning/train_act.py --dataset data/randomized_v1 --episodes 200 --steps 60000 --chunk-size 100 --num-workers 0 --out outputs/act_200
uv run python scripts/eval_policy.py --manifest configs/manifests/so101_randomized_pick_place.yaml --policy lerobot --checkpoint outputs/act_200 --episodes 300 --seed 1000
```

`--episodes` trains on the first N recorded episodes; `--num-workers 0` keeps 200 demonstrations
inside 16 GB of RAM. The whole chunk is played by default; `--policy-arg n_action_steps=N` or
`temporal_ensemble_coeff=C` change that at evaluation time without retraining.

The prototype supports ACT-shaped data and scripted, replay and ACT evaluation. It does
not yet export the standard `LeRobotDataset` format or provide one training entry point
across techniques.
