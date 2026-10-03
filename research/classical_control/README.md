# Classical control

Closed-loop control baselines that do not use a learned model: visual servo,
trajectory generation, and similar techniques.

- `so101_visual_servo.py` — calibrated-camera color-blob visual servo policy
  for SO-101 (`SO101VisualServoPolicy`, `ColorBlobDetector`, `CameraCalibration`).

Registers the `"visual_servo"` policy with `physai.robots.registry` on
import; core never imports this package.

## SO-101 visual servo baseline

The deterministic `visual_servo` policy detects the configured RGB blob in
the front camera, projects its centroid through a pinhole calibration onto
the configured workspace plane, and executes a bounded pick-and-place state
machine through the existing IK and joint-position safety path.

Run one episode:

```bash
uv run python scripts/eval_policy.py --policy visual_servo --episodes 1 --seed 0 --max-steps 400
```

Run the bounded camera-jitter robustness check and save its per-episode
metrics as JSON:

```bash
uv run python scripts/eval_policy.py --policy visual_servo --episodes 20 --seed 0 --max-steps 600 --camera-jitter 0.005 --json-out outputs/visual_servo_20seed_jitter.json
```

The same evaluation runs on a clean Linux runner through the `Research: visual
servo evaluation` workflow
(`.github/workflows/research-visual-servo-eval.yml`). It starts on
pull requests that touch the code the result depends on, and by hand from the
Actions tab once the workflow is on the default branch, with the total episode
count and the jitter as inputs. Software rendering takes about 130 seconds per
episode, so the seeds are split across four parallel jobs and a final job
merges them and fails unless every episode succeeded with no collision,
timeout, or unsafe action. The result table is in the job summary, and the
merged JSON, each shard's JSON, and one recorded episode are uploaded as
artifacts.

The current baseline is **100/100** on seeds 0-99 in MuJoCo and on Isaac Sim
with the same scene and seeds, and 50/50 on the held-out seeds 100-149 (see
[FINDINGS](FINDINGS.md#the-2026-10-03-baseline-hardening-and-difficulty-sweep)).
The CI workflow's all-success check on 20 seeds is expected to pass. A 20-seed
run on `ubuntu-24.04` with OSMesa is not bit-identical to a Windows run, as
expected from floating-point differences between platforms.

### Difficulty sweep and report

`scripts/sweep_difficulty.py` runs the policy over lighting, camera shift and
clutter levels (one `eval_policy.py` process per cell, seeds 100-149 so the
baseline is held out) and writes a Markdown table with the success rate and its
Wilson interval, place error, settling time and failure categories:

```bash
uv run python scripts/sweep_difficulty.py --out-dir outputs/difficulty
uv run python scripts/sweep_difficulty.py --axis camera --episodes 50
uv run python scripts/sweep_difficulty.py --table-only --out-dir outputs/difficulty
```

A single cell can be run by hand with `eval_policy.py --lighting-scale 0.5`,
`--camera-jitter 0.01 --camera-shift-unknown`, `--clutter-count 2` and
`--nominal-physics` (friction and mass stay nominal so only that axis varies).
`--camera-shift-unknown` leaves the policy's calibration at the nominal camera
pose while the camera has moved; without it the policy is told the new pose.
`--policy-arg KEY=VALUE` overrides a policy option (for example
`--policy-arg final_camera=front`) and `--seeds 5,13,28` runs an
explicit seed list.

Isaac Sim takes `--lighting-scale` and `--camera-jitter` with
`--camera-shift-unknown` (clutter is MuJoCo only). Its evaluations are slow and
Isaac sometimes starts without drawing the robot, so run them in shards that
retry on that failure:

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run python scripts/run_sharded_eval.py     --out-dir outputs/eval/isaac_rerun --seed 0 --episodes 100 --shard-size 10     -- --sim isaac --policy visual_servo --max-steps 600
```

`report_evaluation.py` and `compare_evaluations.py` read the merged JSON.

Inspect the policy interactively in the browser — the front and wrist camera
panels are configurable, and a `front:detections`/`wrist:detections` debug
overlay shows the last detected pixel as a crosshair, useful for telling a
detection failure apart from a control failure (see the
[Web Viewer Runbook](../../docs/WEB_VIEWER_RUNBOOK.md#camera-panels)):

```bash
uv run python scripts/run_sim.py --manifest configs/manifests/so101_single_cube_fixed_place.yaml --policy visual_servo --serve --seed 0
```

The default detector targets the red pick cube. The fixed front camera
performs the macro approach; during descent, the moving wrist camera
recalibrates from its current MuJoCo pose and provides a guarded final
alignment correction. After detection, the policy closes the gripper, lifts,
transfers to the configured target, releases, and retreats. Use
`SO101VisualServoPolicy` directly when the target RGB, target pixel, camera
intrinsics, or camera-to-base transform must be changed. The public
calibration uses a right-handed pinhole frame with `+z` forward; MuJoCo's
camera `-z` viewing convention is converted at the adapter boundary. The
fixed front camera can derive its calibration from the environment; the
wrist camera moves with the arm and therefore requires a fresh TF-based
calibration each control tick before it can be used for metric servoing.

The policy exposes `metrics.visual_error_px`, `metrics.ee_error_m`,
`metrics.settled`, and `metrics.failure_reason` separately from task reward.

Isaac Sim closed-loop status (parity ladder tier 4) is tracked in
[ROADMAP.md](../../ROADMAP.md)'s 2E section; the trace evidence behind it is
in [FINDINGS.md](FINDINGS.md).

### Isaac Sim

The same manifest runs on both simulators. Record a video, watch live in the
web viewer (the arm is mirrored; the cube shows in the camera feeds), or
evaluate over seeds and compare with MuJoCo:

```bash
uv run python scripts/run_sim.py --manifest configs/manifests/so101_single_cube_fixed_place.yaml --sim isaac --policy visual_servo --video --camera front
uv run python scripts/run_sim.py --manifest configs/manifests/so101_single_cube_fixed_place.yaml --sim isaac --policy visual_servo --serve
uv run python scripts/eval_policy.py --policy visual_servo --episodes 100 --json-out outputs/mujoco.json
uv run python scripts/eval_policy.py --sim isaac --policy visual_servo --episodes 100 --json-out outputs/isaac.json
uv run python scripts/compare_evaluations.py outputs/mujoco.json outputs/isaac.json
```

`scripts/compare_cameras.py` renders both simulators at one arm pose and
reports colour statistics and what `ColorBlobDetector` finds in each.
Measurements and the causes of the gap are in [FINDINGS.md](FINDINGS.md).
