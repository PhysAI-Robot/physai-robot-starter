# Classical control findings

What was measured and what was tried behind `visual_servo`'s current behavior.
Commands and current results are in [README.md](README.md). Trace logs and
per-attempt narrative are in git history; this file keeps the causes, the numbers
and the negative results.

## Isaac Sim parity: what was wrong

`visual_servo` ran its full phase sequence on Isaac Sim without delivering the
cube. Four causes, in the order they were found:

1. **CLOSE exited at about 21% closed.** The settle check compared the gripper
   position with the *ramping* command (fixed 0.9 rad/s), not the final target.
   MuJoCo's gripper lags the ramp, so the check accidentally waited for the real
   squeeze; Isaac's PhysX drive tracks it with no lag, so CLOSE ended after about
   8 steps and LIFT started on a half-open hand. Fix: settle only once the ramp has
   finished, then accept either tracking the target or a stall
   (`_GRIP_STALL_SETTLE_STEPS = 30`, against 8 for tracking, so a stick-slip pause
   resolves before the arm moves). MuJoCo is unchanged (seed 0 identical, 95/100
   with the same five timeouts before and after).
2. **Isaac gripped with a different contact geometry.** MuJoCo grips with two fitted
   12 x 12 x 6 mm box pads (the jaw mesh's own collision is disabled); Isaac applied
   the pads' friction material to the raw jaw mesh and never built the pads. No
   friction, force or solver setting could make the two comparable.
   `sim.isaac.description.apply_contact_pad_colliders` now builds the same boxes from
   the shared `contact_pads` spec (pad centres 33.9 mm apart at the alignment angle,
   expected 34 mm). With it, the shared default `squeeze_grip=0.15` delivers the cube
   with no per-simulator override.
3. **The static jaw landed on the cube.** The static pad sits about 18 mm toward -x
   of the pinch point and the cube is 28 mm wide, so there is about 1 mm of
   clearance; a few mm of estimate bias put the pad on top. The default
   `grasp_offset_xy=(-0.006, -0.006)` biases the pick to the safe side (found by
   sweeping on Isaac: y = -6 mm gave 6/8 without a hover step against 1/4 at y = 0).
4. **Perception differed between the engines** (found by running identical seeds,
   where Isaac scored 79/100 against MuJoCo's 100/100):
   - `ColorBlobDetector` kept pixels near one bright red, so it dropped the shadowed
     part of the cube and Isaac's deeper shadows made the wrist blob 1700 px against
     3900. It now segments on saturation and hue, giving the same blob to within 1 px
     in both engines.
   - Isaac's default tone mapping washed the cube out (saturation 0.2 against
     MuJoCo's 0.7); far cubes put the hover point outside IK reach. Linear, gamma-free
     output (`sim.isaac.core.configure_render_output`) matches the cube faces to a few
     levels (front face 111, 32, 25 against 112, 33, 26), and light intensities in
     `sim/studio.py` were fitted so table and wrist-cube colours match.

Supporting fixes made on the way: `fx = fy` in both engines (MuJoCo's old
`fx = fy * width / height` scaled x by 4/3 and was a hidden tuning that shifted the
y estimate about 5.5 mm to the safe side); an Isaac wrist-camera calibration
(`camera_calibration("wrist")`, matching the USD camera to about 3 mm); wrist
calibration takes the chosen resolution; `reset()` restores the cube and holds one
step so the first image is fresh; rendering no longer advances Isaac's physics (it
had doubled simulated time per control step, which voided every earlier Isaac
number); float32 joint limits are clipped; the web server runs on its own event loop
because Kit replaces `asyncio.run`.

## Isaac Sim parity: results

Seeds 0-99, same scene and policy defaults, 320 x 240
(`scripts/eval_policy.py` per engine, `scripts/compare_evaluations.py`):

| simulator | success | Wilson 95% | mean steps |
| --- | ---: | --- | ---: |
| MuJoCo | 100/100 | 96-100% | 200 |
| Isaac | 100/100 | 96-100% | 117 |

All 100 seeds agree. A second Isaac run gave 100/100 again (steps within 5 per
seed, mean 116.6 and 116.4, median place error 5.5 and 5.7 mm). The nine far-corner
seeds (x > 0.23, y > 0.11) failed 3 of 90 repeats, about 3% per episode. Other
resolutions: 640 x 480 and 1280 x 720 are 20/20 on MuJoCo and 8/8 on Isaac.

The step gap is one phase: CLOSE takes about 117 steps on MuJoCo (its gripper lags
the ramp, so the settle check rarely passes and CLOSE runs to its 120-step cap) and
about 36 on Isaac. The other phases match to within a step.

## Tried and did not work (Isaac contact tuning, before the geometry fix)

All of these ran against the mismatched pad geometry above, so read them as "not the
cause", not as statements about PhysX.

| Experiment | Result |
| --- | --- |
| Deeper `squeeze_grip` (0.06 and 0.0) on MuJoCo, 50 seeds | 47/50 both, same failing seeds: the torque cap, not the target depth, sets the squeeze once the jaw is jammed |
| `gripper_force_limit` 0.3 to 0.5 N·m on Isaac | grasp-hold test (tier 3) stopped lifting the cube; reverted |
| Per-body PhysX solver iterations on the cube | same failure (cube at rest height 0.0189); reverted |
| Scene-wide `minVelocityIterationCount` 4 | same failure to 7 decimals; reverted |
| Pad friction doubled to 4.0 | CLOSE never settled (jaw jitter exceeded `_GRIP_STALL_EPS`); cube dragged, 0.237 m from target |
| Compliant pad contact (`compliantContactStiffness` and damping) | lifts 2-5 cm but never carries the cube; compliant cube material shoves the cube during DESCEND |
| Half-rate TRANSFER on MuJoCo | 46/50, different failing set; inconclusive, not tried on Isaac |

Lessons: three unrelated "more force or accuracy" changes breaking the same
previously reliable static grasp points to a stability optimum, not an under-tuned
knob; single runs are as noisy as the effects tuned (the same compliance config
gave final distances of 0.145 and 0.247 m), so compare with N >= 5; each Isaac
iteration costs 20 minutes to hours, and a concurrent `uv sync` or `uv run` swaps
the `isaac` extra's packages under a running process, so never run `uv` against this
repo while Isaac runs. Cheap MuJoCo checks only rule a hypothesis in or out when
MuJoCo's contact model exercises it the same way. `SO101VisualServoPolicy`'s
`max_speed` argument is dead code; `max_joint_rate` bounds Cartesian speed.

## Defaults, re-ablated (2026-10-03)

Rule fixed before the runs: keep the defaults unless another setting wins on both
engines by more than the Wilson intervals.

| align | offset (mm) | MuJoCo, seeds 0-99 | mean steps | Isaac, 19 seeds |
| --- | --- | ---: | ---: | ---: |
| on | (-6, -6) (default) | 100/100 | 200.4 | 18/19 |
| off | (-6, -6) | 100/100 | 198.1 | 19/19 |
| on | (0, 0) | 100/100 | 220.3 | 6/19 |
| off | (0, 0) | 100/100 | 210.1 | 15/19 |

The offset matters on Isaac; the hover-until-aligned step did not (44/45 with it,
43/45 without, on the nine far-corner seeds). `align_before_descend` and
`align_tolerance` were removed on 2026-10-04: seeds 100-149 without align matched the
sweep within noise.

## Difficulty sweep (held-out seeds 100-149)

Never used to choose a default. One `eval_policy.py` per cell, friction and mass
nominal, 50 seeds per cell on MuJoCo and 20 (seeds 100-119) on Isaac. These cells
were measured before the 2026-10-04 changes; the section after this one gives the
result after them.

| axis | level | MuJoCo | Wilson 95% | place error mm (median / p90) | Isaac (20 seeds) |
| --- | --- | ---: | --- | ---: | ---: |
| baseline | nominal | 50/50 | 93-100% | 6.0 / 6.5 | 20/20 |
| lighting | x0.5 to x1.6 | 50/50 each | 93-100% | 6.0-6.2 / 6.5-6.6 | 20/20 (x0.5, x1.6) |
| camera shift, policy not told | 5 mm | 50/50 | 93-100% | 6.2 / 10.0 | n/a |
| | 10 mm | 47/50 | 84-98% | 5.9 / 13.8 | 16/20 |
| | 20 mm | 23/50 | 33-60% | 7.5 / 12.7 | 3/20 |
| camera shift, policy told | 20 mm | 47/50 | 84-98% | 5.6 / 6.5 | n/a |
| clutter | 1 / 2 / 4 boxes | 47 / 44 / 44 of 50 | 76-98% | 6.0 / 6.5 | MuJoCo only |

"Camera shift" moves every camera by a uniform offset of up to that many mm per
axis, drawn per episode; "not told" leaves the calibration at the nominal pose (a
bumped or mis-mounted camera). The two engines agree: lighting does not matter and
camera miscalibration does.

Why it fails:
- **Lighting:** nothing to fix; the saturation and hue detector ignores brightness.
- **Camera shift is the wrist camera.** Shifting only the front camera by 20 mm left
  40/40 (the wrist refinement corrects it); only the wrist camera left 27/40. A wrist
  mount error becomes a grasp offset directly, and the jaw clearance is about 1 mm.
  Skipping the wrist refinement (`--policy-arg "final_camera='front'"`) gives 38/40
  shifted but costs about 5 points nominal; `grasp_offset_xy=(-0.009, -0.009)` gave
  28/40. The default stays.
- **Clutter:** the grey-blue boxes are not detected, and the arm is blocked without
  anything noticing. Seeds 100 and 134 ran every phase to RELEASE with the cube moved
  0.000 m; with two boxes, seeds 109 and 124 were stopped by the safety gate in
  TRANSFER because the policy's command kept advancing 0.04 rad per step while the arm
  was held against a box.
- A failure's phase name (`timeout_in_release`) is where the sequence stopped, not
  where the cube was lost; every failure ends 10-27 cm from the target.

Tooling lessons: `info["unsafe_action"]` was never set because the gate raised a
plain `ValueError`; it now raises `SafetyViolation` (still a `ValueError`) and
`eval_policy.py` records it. `failure_reason` is only set for perception errors, so
`report_evaluation.py` derives categories (`unsafe_action`, `collision`,
`feature_not_found`, `finished_not_placed`, `timeout_in_<phase>`) and reports the
Wilson interval and median / p90 place error and settling time. Kit swallows a Python
error's exit status, so retry logic must test for the result file, not the exit code.

## Missed-grasp detection (2026-10-04)

The policy used to run every phase on timers. It now checks from the wrist camera and
joint angles only (the SO-101 has no grasp sensor):
- End of CLOSE: the cube must appear within 25% of the image height of where the pinch
  point projects (held cubes 10-25 px away at 320 x 240, missed 138-174 px or absent).
- End of LIFT: the blob must be within 4% of the image height of where it was at the
  end of CLOSE (a held cube rides with the camera, 0.7 px; one left behind moved 16-23).
- A miss opens the gripper and restarts from APPROACH, at most twice, then stops with
  `failure_reason="grasp_missed"`. A retry is skipped when the remaining steps cannot
  fit one (the failed attempt's length plus 80).

Gripper position and effort alone are not enough (a held cube reads 0.164, an empty
close 0.158, effort at the 0.3 cap in both). The end-of-CLOSE check is absolute, so a
miscalibrated wrist camera biases it; the end-of-LIFT check is relative and does not.

Held-out seeds 100-149, MuJoCo, before (align on, no checks) and now:

| cell | before | now | episodes with a retry | recovered |
| --- | ---: | ---: | ---: | ---: |
| nominal | 50 | 50 | 0 | n/a |
| 1 / 2 / 4 boxes | 47 / 44 / 44 | 48 / 45 / 45 | 2 / 5 / 6 | 0 / 0 / 1 |
| camera shift 10 mm | 47 | 47 | 4 | 1 |
| camera shift 20 mm | 23 | 23 | 24 | 0 |

Seeds 0-99 stay 100/100 with no retry (mean 198 steps). The checks buy a correct
label (in the 20 mm cell 13 of 27 failures are now `grasp_missed` instead of an empty
gripper finishing the sequence), not recovery: 2 of 41 retried episodes succeeded,
and aiming the retry at the wrist-camera estimate recovered 3 of 28, because a miss
here is systematic (a box under the jaw, a biased wrist estimate) and the second
attempt repeats it. The small clutter gains came from removing `align`. Not tried:
approach from the other side, change the pick offset after a miss, move the box.

## Isaac render glitch (open)

Isaac sometimes renders without the robot (no arm, no shadow) for a whole process
while physics works, so camera policies fail for unrelated reasons. It is random per
process, comes in bursts, shows at every resolution, and VRAM is not the cause.
`SO101IsaacEnv.robot_is_rendered()` detects it (render with the robot shown and
hidden, compare); the env refuses to start and `eval_policy --sim isaac` stops at an
episode start, so no robot-less episode is recorded. Run long evaluations in shards
and repeat an aborted one (`scripts/run_sharded_eval.py`).

Over 148 isolated build-and-check starts, 11 glitched (7%). None of these changed the
rate measurably: render mode, sub-frames, 60 warm-up frames, the NVIDIA power mode,
the Fabric delegate, a 90 s cool-down (1 of 5 against 8 of 17 within a minute, p =
0.36), lighting level, or closing the app explicitly. What separates the runs is GPU
memory right after the app starts: all 11 glitches began with 464-587 MiB in use, 136
of 137 clean starts with 1123-1146 MiB, so a start where the renderer has not yet
allocated its memory draws no robot. Not tried: waiting for that memory before
building the env, or clearing the shader cache.

Determinism (negative result): PhysX `enableEnhancedDeterminism` did not make runs
identical (the same seed gave 128, 134 and 600 steps without it and 600, 133, 134
with it), so it was reverted. Run-to-run variation is small next to the effects in
the sweep.
