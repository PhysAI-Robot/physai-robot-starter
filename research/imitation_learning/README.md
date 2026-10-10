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

### Diagnosing a gap between simulators

The tools behind the [M4 findings](FINDINGS.md#sim-to-sim-gap-on-the-randomized-task-study-1-m4).
All evaluation-time edits change only what the policy is shown, not the robot:

```bash
# one image region edited (bg_dim, bg_gray, bg_flat, table_dim, cube_gain, arm_tint,
# isaac_like, blank_front, blank_wrist, blur, blur_front, blur_wrist), or the gripper reading
# replaced by the other simulator's behaviour (gripper_track, gripper_lag)
uv run python scripts/eval_policy.py --manifest configs/manifests/so101_randomized_pick_place.yaml --policy lerobot --checkpoint outputs/act_200 --seed 900 --policy-arg image_edit=bg_flat
uv run python scripts/eval_policy.py ... --policy-arg state_edit=gripper_track

# first-frame swap: the step-0 camera frames replaced by Isaac's render of the same seed
# (first_all, first_front, first_wrist, first_arm, first_cubedisc, first_table, first_background).
# data/m5_first/{mujoco,isaac} come from running eval_policy.py --policy constant --max-steps 2
# --save-dataset once per simulator over the same seeds
uv run python scripts/eval_policy.py ... --policy-arg image_edit=first_all --policy-arg first_frames=data/m5_first --policy-arg first_seed=900

# the same demonstration actions replayed on Isaac Sim, recording Isaac's frames as a
# training dataset (the actions are identical, so only the pictures differ)
OMNI_KIT_ACCEPT_EULA=YES uv run python scripts/eval_policy.py --manifest configs/manifests/so101_randomized_pick_place.yaml --sim isaac --policy replay --dataset data/randomized_v1 --seed 0 --episodes 100 --save-dataset data/randomized_isaac_v1

# train on frames from both simulators, or with random colour changes
uv run python research/imitation_learning/train_act.py --dataset data/randomized_v1 data/randomized_isaac_v1 --chunk-size 100 --steps 120000 --num-workers 0 --out outputs/act_mixed
uv run python research/imitation_learning/train_act.py --dataset data/randomized_v1 --color-jitter ...

# sensitivity of any policy to lighting, camera shift and clutter
uv run python scripts/sweep_difficulty.py --policy lerobot --checkpoint outputs/act_200 --manifest configs/manifests/so101_randomized_pick_place.yaml --seed 1000 --episodes 100
```

Several `--dataset` folders are joined; their policy-sized frames are cached once in
`.act_cache_<image_size>/` beside each dataset and read memory-mapped, so 400 episodes need
disk and page cache rather than 8 GB of RAM. Replays on Isaac Sim need `--sim isaac`; stop
other GPU work first and use the project's own `.venv` python while it runs.

The prototype supports ACT-shaped data and scripted, replay and ACT evaluation. It does
not yet export the standard `LeRobotDataset` format or provide one training entry point
across techniques.
