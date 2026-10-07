# Classical control findings

What was measured behind `visual_servo`'s current behavior: causes, final numbers and one
line per failed experiment. Commands and current results are in [README.md](README.md);
the trace is in git history.

## Isaac Sim parity: what was wrong

`visual_servo` ran its full phase sequence on Isaac without delivering the cube. Four causes:

1. **CLOSE exited at about 21% closed.** The settle check compared the gripper with the
   *ramping* command, not the final target. MuJoCo's gripper lags the ramp, so the check
   accidentally waited for the real squeeze; Isaac's PhysX drive tracks it, so LIFT started on
   a half-open hand. Fix: settle only once the ramp has finished, then accept tracking the
   target or a stall (`_GRIP_STALL_SETTLE_STEPS = 30`).
2. **Different contact geometry.** MuJoCo grips with two fitted 12 x 12 x 6 mm box pads (the
   jaw mesh's collision is disabled); Isaac applied the friction material to the raw jaw mesh.
   `sim.isaac.description.apply_contact_pad_colliders` now builds the same boxes from the
   shared `contact_pads` spec, and the shared `squeeze_grip=0.15` works with no per-engine
   override.
3. **The static jaw landed on the cube.** The static pad sits about 18 mm toward -x of the
   pinch point and the cube is 28 mm wide, leaving about 1 mm clearance. The default
   `grasp_offset_xy=(-0.006, -0.006)` biases the pick to the safe side (Isaac sweep: y = -6 mm
   gave 6/8 against 1/4 at y = 0).
4. **Perception differed.** Identical seeds scored 79/100 on Isaac and 100/100 on MuJoCo.
   `ColorBlobDetector` kept pixels near one bright red and lost the shadowed cube faces, so it
   now segments on saturation and hue (same blob to within 1 px). Isaac's default tone mapping
   washed the cube out (saturation 0.2 against 0.7), so output is linear and gamma-free
   (`sim.isaac.core.configure_render_output`) and light intensities in `sim/studio.py` were fitted.

Supporting fixes: `fx = fy` in both engines (the old MuJoCo `fx = fy * width / height` hid a
5.5 mm bias), an Isaac wrist-camera calibration, `reset()` holds one step so the first image is
fresh, and rendering no longer advances Isaac's physics (it had doubled simulated time per
control step, voiding every earlier Isaac number).

## Isaac Sim parity: results

Seeds 0-99, same scene and defaults, 320 x 240 (`eval_policy.py` per engine, then
`compare_evaluations.py`):

| simulator | success | Wilson 95% | mean steps |
| --- | ---: | --- | ---: |
| MuJoCo | 100/100 | 96-100% | 200 |
| Isaac | 100/100 | 96-100% | 117 |

All 100 seeds agree, and a second Isaac run gave 100/100 again. The nine far-corner seeds
failed 3 of 90 repeats on Isaac (about 3% per episode). 640 x 480 and 1280 x 720 are 20/20 on
MuJoCo and 8/8 on Isaac. The step gap is the CLOSE phase (about 117 steps on MuJoCo, which runs
to its 120-step cap because its gripper lags the ramp, against about 36 on Isaac).

## Tried and did not work (Isaac contact tuning, before the geometry fix)

Run against the mismatched pad geometry, so read these as "not the cause":

- Deeper `squeeze_grip` (0.06, 0.0) on MuJoCo, 50 seeds: 47/50 both; the torque cap sets the squeeze.
- `gripper_force_limit` 0.3 to 0.5 N·m, per-body PhysX solver iterations, scene-wide
  `minVelocityIterationCount` 4: each stopped the grasp-hold test lifting the cube; reverted.
- Pad friction doubled to 4.0: CLOSE never settled (jaw jitter above `_GRIP_STALL_EPS`).
- Compliant pad contact: lifts 2-5 cm but never carries the cube.
- Half-rate TRANSFER on MuJoCo: 46/50, different failing set; inconclusive.

Lessons: several "more force or accuracy" changes breaking the same grasp point to a stability
optimum, not an under-tuned knob; compare with N >= 5 (one config gave final distances of 0.145
and 0.247 m); never run `uv` against this repo while Isaac runs, it swaps the extra's packages
under the process.

## Defaults, re-ablated

Rule fixed before the runs: keep the defaults unless another setting wins on both engines by
more than the Wilson intervals.

| align | offset (mm) | MuJoCo, seeds 0-99 | Isaac, 19 seeds |
| --- | --- | ---: | ---: |
| on | (-6, -6) (default) | 100/100 | 18/19 |
| off | (-6, -6) | 100/100 | 19/19 |
| on | (0, 0) | 100/100 | 6/19 |
| off | (0, 0) | 100/100 | 15/19 |

The offset matters on Isaac; the hover-until-aligned step did not, so `align_before_descend`
and `align_tolerance` were removed (seeds 100-149 without align matched the sweep).

## Difficulty sweep (held-out seeds 100-149)

Never used to choose a default. One `eval_policy.py` per cell, 50 seeds per cell on MuJoCo and
20 (seeds 100-119) on Isaac, measured before missed-grasp detection:

| axis | level | MuJoCo | Wilson 95% | place error mm (median / p90) | Isaac (20 seeds) |
| --- | --- | ---: | --- | ---: | ---: |
| baseline | nominal | 50/50 | 93-100% | 6.0 / 6.5 | 20/20 |
| lighting | x0.5 to x1.6 | 50/50 each | 93-100% | 6.0-6.2 / 6.5-6.6 | 20/20 (x0.5, x1.6) |
| camera shift, policy not told | 5 mm | 50/50 | 93-100% | 6.2 / 10.0 | n/a |
| | 10 mm | 47/50 | 84-98% | 5.9 / 13.8 | 16/20 |
| | 20 mm | 23/50 | 33-60% | 7.5 / 12.7 | 3/20 |
| camera shift, policy told | 20 mm | 47/50 | 84-98% | 5.6 / 6.5 | n/a |
| clutter | 1 / 2 / 4 boxes | 47 / 44 / 44 of 50 | 76-98% | 6.0 / 6.5 | MuJoCo only |

"Camera shift" moves every camera by a uniform offset of up to that many mm per axis; "not told"
leaves the calibration at the nominal pose. Both engines agree: lighting does not matter and
camera miscalibration does.

- **Lighting:** the saturation and hue detector ignores brightness.
- **Camera shift is the wrist camera.** Shifting only the front camera by 20 mm left 40/40; only
  the wrist camera left 27/40, because a wrist mount error becomes a grasp offset directly
  against about 1 mm of jaw clearance. Skipping the wrist refinement gives 38/40 shifted but
  costs about 5 points nominal; `grasp_offset_xy=(-0.009, -0.009)` gave 28/40. The default stays.
- **Clutter:** the grey-blue boxes are not detected and nothing notices the arm is blocked
  (cube moved 0.000 m through every phase on seeds 100 and 134; the safety gate stopped seeds
  109 and 124 in TRANSFER while the command kept advancing 0.04 rad per step).
- A failure's phase name (`timeout_in_release`) is where the sequence stopped, not where the
  cube was lost; every failure ends 10-27 cm from the target.

Tooling: the gate raises `SafetyViolation` (a `ValueError`) so `eval_policy.py` records
`unsafe_action`; `report_evaluation.py` derives the failure categories (`unsafe_action`,
`collision`, `feature_not_found`, `finished_not_placed`, `timeout_in_<phase>`); Kit swallows a
Python error's exit status, so retry logic must test for the result file.

## Missed-grasp detection

The policy checks from the wrist camera and joint angles only (the SO-101 has no grasp sensor):
at the end of CLOSE the cube must appear within 25% of the image height of the pinch point's
projection (held 10-25 px away, missed 138-174 px or absent), and at the end of LIFT the blob
must be within 4% of where it was (held 0.7 px, left behind 16-23). A miss opens the gripper and
restarts from APPROACH at most twice, then stops with `failure_reason="grasp_missed"`. Gripper
position and effort alone are not enough (held 0.164, empty 0.158, effort at the 0.3 cap in
both).

Held-out seeds 100-149, MuJoCo, before and now: nominal 50 to 50; 1 / 2 / 4 boxes 47 / 44 / 44 to
48 / 45 / 45; camera shift 10 mm 47 to 47; 20 mm 23 to 23 (24 episodes retried, 0 recovered).
Seeds 0-99 stay 100/100 with no retry. The checks buy a correct label (13 of 27 failures in the
20 mm cell are now `grasp_missed`), not recovery: 2 of 41 retried episodes succeeded, and aiming
the retry at the wrist estimate recovered 3 of 28, because a miss here is systematic (a box
under the jaw, a biased wrist estimate) and the second attempt repeats it. Not tried: approach
from the other side, change the pick offset after a miss, move the box.

## Isaac render glitch (open)

Isaac sometimes renders without the robot for a whole process while physics works, so camera
policies fail for unrelated reasons. `SO101IsaacEnv.robot_is_rendered()` detects it; the env
refuses to start and `eval_policy --sim isaac` stops at an episode start. Run long evaluations
in shards and repeat an aborted one (`scripts/run_sharded_eval.py`). Over 148 isolated starts
11 glitched (7%). Render mode, sub-frames, warm-up frames, the power mode, the Fabric delegate,
a cool-down, lighting and closing the app explicitly did not change the rate. What separates the
runs is GPU memory right after the app starts: all 11 glitches began with 464-587 MiB in use, 136
of 137 clean starts with 1123-1146 MiB. Not tried: waiting for that memory before building the
env, or clearing the shader cache. PhysX `enableEnhancedDeterminism` did not make runs identical
(same seed gave 128, 134 and 600 steps), so it was reverted.
