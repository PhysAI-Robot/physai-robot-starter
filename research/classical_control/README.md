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

The same evaluation runs on a clean Linux runner through the `Visual servo
evaluation` workflow (`.github/workflows/visual-servo-eval.yml`). It starts on
pull requests that touch the code the result depends on, and by hand from the
Actions tab once the workflow is on the default branch, with the total episode
count and the jitter as inputs. Software rendering takes about 130 seconds per
episode, so the seeds are split across four parallel jobs and a final job
merges them and fails unless every episode succeeded with no collision,
timeout, or unsafe action. The result table is in the job summary, and the
merged JSON, each shard's JSON, and one recorded episode are uploaded as
artifacts.

The current baseline is **95/100** with 5 timeouts and no collisions or unsafe
actions (see [Results](../scripted_experts/README.md#results)); it was 100%
before the fingertip pad refit (`fad205d`), and the timeouts are not yet
diagnosed. Seeds 13 and 15 time out without camera jitter, so expect the
workflow's all-success check on 20 seeds to fail until this is fixed. Before
the refit, a 20-seed run on `ubuntu-24.04` with OSMesa reproduced 20/20; it was
not bit-identical to a Windows run (17 of 20 seeds took the same number of
steps, the other three differed by one), as expected from floating-point
differences between platforms.

Inspect the policy interactively in the browser — the front and wrist camera
panels are configurable, and a `front:detections`/`wrist:detections` debug
overlay shows the last detected pixel as a crosshair, useful for telling a
detection failure apart from a control failure (see the
[Web Viewer Runbook](../../docs/WEB_VIEWER_RUNBOOK.md#camera-panels)):

```bash
uv run python scripts/run_sim.py --config configs/tasks/so101/pick_place.yaml --policy visual_servo --serve --seed 0
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
