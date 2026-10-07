# Classical control

Closed-loop control baselines that do not use a learned model: visual servo and
similar techniques.

- `so101_visual_servo.py`: calibrated-camera colour-blob visual servo for SO-101
  (`SO101VisualServoPolicy`, `ColorBlobDetector`, `CameraCalibration`).

It registers the `"visual_servo"` policy with `physai.robots.registry` on import; core
never imports this package.

## SO-101 visual servo baseline

The policy detects the configured RGB blob in the front camera, projects its centroid
through a pinhole calibration onto the workspace plane, and runs a bounded
pick-and-place state machine through the existing IK and joint-position safety path.
The fixed front camera does the macro approach; during descent the wrist camera
recalibrates from its current pose and gives a guarded final alignment. After the
grasp it lifts, transfers, releases and retreats, and it checks from the wrist camera
and joint angles that the cube is held (a miss retries twice, then stops with
`grasp_missed`). Use `SO101VisualServoPolicy` directly to change the target colour,
target pixel, intrinsics or camera-to-base transform. The public calibration is a
right-handed pinhole frame with `+z` forward; MuJoCo's `-z` convention is converted at
the adapter boundary. The wrist camera moves with the arm, so its calibration is
recomputed every control tick. The policy exposes `metrics.visual_error_px`,
`metrics.ee_error_m`, `metrics.settled` and `metrics.failure_reason` separately from
reward.

Current results are in the
[results table](../scripted_experts/README.md#results); the measurements and failure
analysis behind them are in [FINDINGS.md](FINDINGS.md).

```bash
uv run python scripts/eval_policy.py --policy visual_servo --episodes 1 --seed 0 --max-steps 400
uv run python scripts/eval_policy.py --policy visual_servo --episodes 20 --seed 0 --max-steps 600 --camera-jitter 0.005 --json outputs/visual_servo_20seed_jitter.json
uv run python scripts/run_sim.py --manifest configs/manifests/so101_single_cube_fixed_place.yaml --policy visual_servo --serve --seed 0
```

The same 20-seed evaluation runs in CI as `Research: visual servo evaluation`
(`.github/workflows/research-visual-servo-eval.yml`): on pull requests that touch the code
it depends on, and by hand from the Actions tab. Software rendering takes about 130 s
per episode, so seeds are split across four parallel jobs and a final job merges them
and fails unless every episode succeeded with no collision, timeout or unsafe action. A
20-seed run on `ubuntu-24.04` with OSMesa is not bit-identical to a Windows run
(floating-point differences). In the browser, the camera panels are configurable and a
`front:detections` / `wrist:detections` overlay shows the last detected pixel, which
separates a detection failure from a control failure
([runbook](../../docs/WEB_VIEWER_RUNBOOK.md#browser-controls)).

### Difficulty sweep and report

`scripts/sweep_difficulty.py` runs the policy over lighting, camera-shift and clutter
levels (one `eval_policy.py` process per cell, seeds 100-149 so the baseline is held
out) and writes a Markdown table with the success rate, Wilson interval, place error,
settling time and failure categories:

```bash
uv run python scripts/sweep_difficulty.py --out outputs/difficulty
uv run python scripts/sweep_difficulty.py --axis camera --episodes 50
uv run python scripts/sweep_difficulty.py --table-only --out outputs/difficulty
```

A single cell runs by hand with `eval_policy.py --lighting-scale 0.5`,
`--camera-jitter 0.01 --camera-shift-unknown`, `--clutter-count 2` and
`--nominal-physics` (friction and mass stay nominal so only that axis varies).
`--camera-shift-unknown` leaves the policy's calibration at the nominal pose while the
camera has moved. `--policy-arg KEY=VALUE` overrides a policy option (for example
`--policy-arg final_camera=front`) and `--seeds 5,13,28` runs an explicit seed list.
`--video` writes one video per episode under `--out` (default `outputs`, in `videos/`,
named `<simulator>_<robot>_<policy>_seed<seed>.mp4`, with `_02`, `_03` added on a
repeat; `--name` replaces the prefix and `run_sim.py` uses the same names),
`--video failures` only for failed episodes, and `--camera` picks the camera (default
the robot's first camera, `front`). `report_evaluation.py` and `compare_evaluations.py` read the merged JSON.

## Isaac Sim

The same manifest runs on both simulators (`uv sync --extra isaac`, then set
`OMNI_KIT_ACCEPT_EULA=YES` yourself; see [DECISIONS.md D](../../docs/DECISIONS.md#d-isaac-sim-is-an-optional-peer-engine)):

```bash
uv run python scripts/run_sim.py --manifest configs/manifests/so101_single_cube_fixed_place.yaml --sim isaac --policy visual_servo --video --camera front
uv run python scripts/run_sim.py --manifest configs/manifests/so101_single_cube_fixed_place.yaml --sim isaac --policy visual_servo --serve
uv run python scripts/eval_policy.py --policy visual_servo --episodes 100 --json outputs/mujoco.json
uv run python scripts/eval_policy.py --sim isaac --policy visual_servo --episodes 100 --json outputs/isaac.json
uv run python scripts/compare_evaluations.py outputs/mujoco.json outputs/isaac.json
```

On Isaac, `--lighting-scale` and `--camera-jitter` (with `--camera-shift-unknown`) work;
clutter is MuJoCo only. Evaluations are slow and Isaac sometimes starts without drawing
the robot, so run them in shards that retry on that failure:

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run python scripts/run_sharded_eval.py --out outputs/eval/isaac_rerun --seed 0 --episodes 100 --shard-size 10 -- --sim isaac --policy visual_servo --max-steps 600
```

`scripts/compare_cameras.py` renders both simulators at one arm pose and reports colour
statistics and what `ColorBlobDetector` finds in each.
