# Project Roadmap: physai-robot-starter

`physai-robot-starter` is an open-source starter kit for embodied AI and robotics research. It provides stable robot, task, observation, and action contracts so the same policy can be evaluated across backends: direct MuJoCo, ROS2 + MuJoCo, and (in the future) real robots.

## Supported robots

| Robot | Status | Notes |
| --- | --- | --- |
| SO-101 | Supported, **current focus** | All new research work goes here |
| TurtleBot4 | Supported, maintained | Existing navigation code and tests stay green; no new work |
| Others | Planned | Added later through the robot adapter interface |

**Current focus robot:** SO-101

## Scope

- **Simulation only for now.** No hardware work is planned; the architecture stays open for it.
- **New robots** are added only after the current focus robot has a solid report (see Phase 3 and 4).

## Out of scope / limitations

| Robot | Out of scope / limitation | Status |
| --- | --- | --- |
| SO-101 | Transparent or thin real-world objects (glass, ballpoint pens): hard for simulated contact and vision, and for the low-cost gripper | Revisit after the v0.2 report |
| TurtleBot4 | Research benchmark (task ladder, baselines, learned policies) | To be implemented after the SO-101 v0.2 report |

## Research story

Phase 2 answers one question: **why is learning needed for manipulation, and how far does it get us?** Three methods are compared with the same seeds, tasks, and metrics:

| Method | Perception | Role |
| --- | --- | --- |
| Scripted expert | Privileged simulator state | Intended upper bound and demonstration teacher — 100% on the single-cube task, matching the camera-only baseline; see the Phase 2.0 finding |
| Classical vision + state machine | Camera only | What hand-engineered perception and control can do without cheating |
| ACT (imitation learning) | Camera + proprioception | Does learning from pixels close the gap or extend beyond the classical limits? |

All methods are evaluated on a difficulty sweep (lighting, camera shift, clutter, distractors, occlusion). The expected result is that the classical pipeline degrades as perception gets harder; the curves are the evidence, not the narrative.

## Current status

**Phase 1 and the training bridge are complete. Phase 2 is the current focus.**
`[x]` = deliverable exists with focused test coverage. `[ ]` = planned, missing, or partial.

```
Phase 1 (done) -> Bridge (done) -> Phase 2: benchmark -> classical baseline -> ACT -> backend study -> v0.2 report
                                                                                                          |
                                                                       Phase 3 and 4: not in focus yet
```

---

## Phase 1: Foundation and ROS2 contract (complete)

- [x] Capability-aware `Observation -> Action` contracts, robot registry, unit/frame validation, deterministic resets, seeded regression coverage.
- [x] SO-101 MuJoCo baseline: scripted single-cube pick-and-place. Measured
      2026-09-21 at **100% over 100 seeds** after the orientation-constraint
      fix; see the Phase 2.0 finding below.
- [x] SO-101 ROS2 bridge: joint, gripper, camera, TF, teleoperation, `rclpy` acceptance coverage.
- [x] SO-101 FK, Jacobian, numerical IK, Cartesian targeting, joint-limit and collision safety validation.
- [x] Seeded domain randomization (physics, visuals, cameras, clutter) with deterministic baseline preserved.
- [x] TurtleBot4 navigation baseline and ROS2/Nav2 path (maintained only).
- [x] Phase 1 runs without Phase 2+ dependencies (LeRobot, VLA).

## Training bridge (complete)

- [x] Canonical `ObservationSpec` / `ActionSpec` schemas and robot-owned training contracts.
- [x] Gymnasium adapter routed through the existing safety gate.
- [x] Explicit SO-101 action layout shared by policies, recorder, replay, ROS2 conversion, and datasets.
- [x] Versioned dataset metadata; checkpoint metadata with compatibility validation.
- [x] Shared evaluation reports (success, collision, timeout, unsafe action, reward, held-out seeds).

---

## Phase 2: Benchmark, baselines, and learning (current focus)

### 2.0 Task ladder and capability report

Define what the SO-101 can do, with one fixed-seed evaluation per level.

| Level | Task | Tests |
| --- | --- | --- |
| T0 | Single cube, fixed setup | Scripted 100% over 100 seeds — done |
| T1 | Single cube, randomized pose / color / lighting / camera | Perception robustness |
| T2 | Cube size variation | Gripper aperture limits |
| T3 | Shape variation (cylinder, sphere, prism) | Grasp difficulty |
| T4 | Multi-object sorting: 3 colored cubes into 3 bins, then scale toward 9 | Multi-step planning and perception |
| T5 (stretch) | Clutter, distractors, occlusion, stacking | Hard perception and contact |

**Resolved finding (2026-09-21, opened 2026-09-20).** The scripted expert was
far weaker than the `20/20` previously recorded. Root cause and fix:

`ExpertConfig.approach_dir` defaulted to `None` and `_solve()` passed it to
`ArmKinematics.ik_pinch()` explicitly, which **overrode** `ik_pinch()`'s own
`TOP_DOWN` default — so the scripted expert solved IK with no wrist-orientation
term at all. The redundant arm could satisfy the pinch-point position target
with the wrist at an arbitrary tilt; the moving jaw sits farther from the
wrist than the static one, so a few degrees of unwanted tilt was enough to
swing it clear of the cube at squeeze time (pads measured 20-50 mm off and
17-20 mm high on failing episodes, versus ~16 mm and a few mm on successes).
`visual_servo` never passed `approach_dir`, so it always got the `TOP_DOWN`
constraint — which is why "the camera baseline is 100% cheating knows the
coordinates but still can't beat it" was the right question to ask. Setting
`ExpertConfig.approach_dir` to default to `TOP_DOWN` closed the gap with no
other change (`approach_rate` stayed at its original 0.5).

The orientation fix alone brought single-cube success to 100% and sorting to
90% (from 46%). Sorting's remaining gap was then traced with
`mujoco.mj_contactForce`: the cube was grasped correctly and lifted, but the
pad normal force at `gripper_grip=0.19` measured only **0.1-1 N** (the
actuator can produce up to ~2.94 N), so the grip gradually let go during
TRANSFER's acceleration and the cube fell back to the table. The cause is
`gripper_grip=0.19` sitting barely past `gripper_touch=0.21` — with a
position-controlled jaw, squeeze force against a rigid cube comes entirely
from how far past first contact the commanded position sits, and 0.02 of
normalized aperture was not far enough. Deepening the squeeze target to
`gripper_grip=0.06` raised the measured contact force and closed most of the
gap, with **no other change** (pad friction, pad size, and a LIFT settle dwell
were all tried and made things worse or did nothing — see below).

Measured over held-out seeds with Wilson 95% intervals, original vs. final:

| Policy | Variant | Seeds | Success (original) | Success (final) |
| --- | --- | --- | --- | --- |
| `scripted` | Single cube | 100 | 59% [49%, 68%] | **100% [98%, 100%]** (300 seeds) |
| `scripted` | Sorting | 50 | 46% [33%, 60%] | **98% [97%, 99%]** (900 seeds) |
| `visual_servo` | Single cube | 20 | 100% [84%, 100%] | 100% (unchanged) |

The single-cube result matches `visual_servo` exactly (200/200 through the
production `scripts/eval_policy.py` path, zero collisions/timeouts/unsafe
actions across every validation run), so the scripted expert is now a valid
upper bound and demonstration teacher.

**A third sorting-specific mechanism, found after the first two fixes.** With
the grip force fixed, sorting still sat at 94% while pick-place — same robot,
same expert code, same physics — was at 100%. The two tasks differ only in
scene layout: sorting places three cubes 6 cm apart; pick-place has one cube
on an otherwise empty table. Bucketing sorting failures by which of the three
spawn bands held the target cube showed a 6x spread (middle band, flanked by
a neighbor on both sides, failed 12.6% of the time vs. 1.9% and 4.0% for the
two edge bands, which only have one neighbor each) — ruling out workspace
reach and pointing at the neighbors specifically. Direct contact logging
confirmed the mechanism: the arm's pads touch a neighboring cube in every
single middle-band episode, success or failure alike, during APPROACH's wide
lateral sweep (jaws command `gripper_open`, spanning 79 mm against 28 mm
cubes 6 cm apart). That contact is usually harmless, but `_grasp_xy` is
locked once at the start of APPROACH and never revisited, so on the episodes
where the nudge lands on the *target* cube, every later phase keeps aiming at
a now-stale position: cube drift from the locked aim point averaged 6 mm
(median 0 mm) in successes vs. 79 mm (median 70 mm) in failures. Refreshing
`_grasp_xy` once — at the APPROACH-to-DESCEND handoff, not continuously —
took sorting from 94% to 98% over 900 held-out seeds across three seed
ranges, with pick-place (nothing nearby to nudge) unaffected.

A companion idea — narrowing the APPROACH/DESCEND aperture to just clear the
cube (11 mm margin at `gripper_grip=0.25` vs. 51 mm at the default
`gripper_open=0.55`) — was tried on the reasoning that a narrower jaw would
be less likely to reach a neighbor in the first place. It measurably made
things worse (94% -> 81%): removing the wide-open slack made the jaws far
more likely to clip the *target* cube itself during approach, given the
system's existing positioning precision, which is a worse failure than the
one it was meant to prevent. Not applied; `gripper_open` stays `0.55` for
APPROACH/DESCEND.

Two more things tried during the sorting work did not survive validation and
were reverted:

- **A pad-geometry-aware IK offset regressed and was reverted.** Suspecting
  `PINCH_OFFSET` (a fixed constant, measured accurate to ~3 mm near a 0.19
  aperture but drifting past 10 mm by 0.06) was still costing accuracy,
  `ArmKinematics` was extended to compute the offset from the live pad geoms
  instead. It measurably made things worse (98.7% -> ~85% on a 150-seed
  check): the geometric midpoint of the pad geoms' origins is not the same
  reference point the calibrated constant represents, so "exact" was exact
  for the wrong target. Fully reverted; `kinematics.py` is unchanged from
  before this investigation.
- **`gripper_force_limit` (a pre-existing `EnvConfig` field, default `0.3`)
  turned out to be load-bearing for a different policy.** With the deep
  0.06 squeeze, contact force is already low without any cap (~0.1-1 N
  measured via `mj_contactForce`), and a fair 300-seed sorting comparison
  showed every tested cap value (0.3-2.0 N) doing 2-5 points *worse* than no
  cap at all (96.0% uncapped vs 92.7-94.7% capped). But `visual_servo` uses
  its own, shallower, hardcoded 0.19 aperture, and disabling the cap broke
  `test_visual_servo_pick_place_settles_from_multiple_seeds` outright: at
  that shallow aperture, uncapped force can reach the actuator's full ~2.94 N
  rating against a rigid cube and destabilize the contact, which is exactly
  what the cap exists to prevent. The shared default was kept at `0.3` to
  keep `visual_servo` correct.
- **Requiring both pads in contact before accepting a grasp** (tightening
  `_cube_grasped()`, which currently accepts either pad touching) was tried
  on the theory that a one-sided contact — measured directly on one failing
  seed as 0 N on one pad, 4.2 N on the other — should trigger a retry before
  ever lifting. It made sorting *worse* (94% -> 91.7%): the stricter check
  also retries some contacts that would have held fine, and each retry
  re-runs the full APPROACH-to-SQUEEZE cycle, so the added retries burned
  step budget into new timeouts faster than they fixed weak grasps.

What remains after the grasp-refresh fix (98% over 900 seeds) is believed to
be a small, genuine task/physics floor: a handful of seeds still fail
regardless of `gripper_grip` depth (0.06-0.19) or the force cap, and
`visual_servo` -- independent grip logic, unaffected by `ExpertConfig` --
fails on some of the same ones. No retry-on-drop was added anywhere in this
investigation.

Two things were established while investigating, and they still shape 2.0:

1. **A real bug was fixed along the way.** Grasp detection looked up a
   hardcoded `cube_geom`, which does not exist in the sorting scene (its
   cubes are `cube_red`, `cube_blue`, `cube_yellow`). The lookup returned -1,
   no grasp was ever detected, and the expert retried until timeout. Covered
   by a regression test.
2. **The 20-seed protocol cannot measure this policy.** The same
   configuration scored 45% on seeds 0-19 and 60% on seeds 100-149 before the
   fix. The historical `20/20` was partly sampling luck. This is why the
   ≥100-seed rule below matters.

Deliverables:

- [x] Diagnose the scripted-expert timeouts and restore a high single-cube success rate.
- [x] Diagnose and reduce the sorting transfer-slip and neighbor-clutter failures (46% -> 98%).
- [ ] Task definitions and scripted experts for T1-T4 (T5 stretch), each with fixed seeds.
- [ ] `scripts/capability_report.py`: reachable workspace, min/max graspable size, placement repeatability.
- [ ] Trajectory-quality metrics for the scripted expert: completion time, path length, jerk, joint-limit margin, peak speed.
- [ ] Compare the current scripted trajectory with a smoother variant (for example minimum-jerk). Smooth, consistent demonstrations matter for imitation learning; "optimal" is not required.

Definition of done:

1. The scripted expert reaches about 100% on T0-T4 (lower confidence bound reported). If it cannot, the task or the robot limit is documented as a finding.
2. T0 and T1 are re-evaluated on **at least 100 seeds**, not 20.
3. The capability report answers what sizes, shapes, and positions the SO-101 can handle in simulation.

### 2A Classical vision baseline (required)

A camera-only state-machine pipeline (color segmentation or fiducials, pose estimate, approach, grasp, place), all actions passing through the safety layer. This is the "before learning" reference.

- [x] Perception module without simulator ground truth (`ColorBlobDetector` in `src/physai/robots/so101/visual_servo.py`; reads camera calibration only, never object pose).
- [x] State machine covering approach, grasp, lift, place, and recovery on failure (`SO101VisualServoPolicy`, registered as the `visual_servo` policy).
- [x] All actions pass the safety layer: `SafetyController` now gates the direct-MuJoCo path inside `DirectMuJoCoAdapter`, not only the ROS2 and Gymnasium paths.
- [ ] Report position error, settling time, and categorized failure reasons.
- [ ] Give it a fair tuning effort; it must not be a strawman.

Definition of done: evaluated on T0-T4 and the difficulty sweep with the shared protocol, with its failure modes documented.

### 2B Imitation learning with ACT

Train from scripted-expert demonstrations. The configuration must fit a single 6 GB laptop GPU.

- [ ] Fix and document the training configuration (image size, chunk size, batch size, precision).
- [ ] `scripts/collect_demos.py`: export standard `LeRobotDataset` format (currently a LeRobot-shaped `.npz`, partial).
- [ ] `scripts/train_act.py`: fixed seeds, logged loss curves, checkpoint metadata.
- [ ] `scripts/eval_policy.py`: closed-loop held-out evaluation with success, collision, timeout, and unsafe-action counts.
- [ ] At least two ablations (for example number of demonstrations, camera views, randomization on/off).

Definition of done: 50-100 demonstrations per task, reproducible training, evaluation on the shared protocol for T0-T4, and categorized failure analysis.

### 2C Backend comparison study

Run the same checkpoint through direct MuJoCo and through the ROS2 bridge to measure the effect of the integration layer, with no hardware needed.

- [ ] ROS2-in-the-loop evaluation path in `scripts/eval_policy.py`.
- [ ] Metrics per backend: success, end-to-end latency, effective control rate, action deviation, unsafe-action rejections.
- [ ] Optional controlled perturbations: added latency, reduced control rate, camera noise.

Definition of done: identical seeds on both backends, a results table, and the main causes of any gap identified.

### 2D Report and release (v0.2)

- [ ] 2-4 page report in `docs/`: setup, task ladder, three-method comparison, difficulty sweep, ablations, backend comparison, failure analysis, limitations.
- [ ] Tag `v0.2` with the exact configs, seeds, and checkpoints to reproduce every reported number.

### Evaluation protocol (shared by all methods)

- Held-out seeds: at least 50 per task and condition (at least 100 for T0/T1).
- Report success rate with a 95% confidence interval (for example Wilson).
- Metrics: success, collision, timeout, unsafe action, completion time.
- Difficulty axes: lighting, camera shift, clutter, distractors, occlusion.

### Optional after the report

- State-based deep RL (PPO or SAC) as a further comparison. Vision-based RL only if the state-based version works.

---

## Phase 3 and Phase 4: not in focus yet

**Not in focus yet.** Listed only so the architecture keeps room for them.

- **Phase 3, language and planning:** only the planner *contract* and its scripted backend remain (`physai.planner`, `scripts/plan_task.py`). The SmolVLM and Claude backends were removed in the 2026-09-20 cleanup — they were untested, unrunnable without absent dependencies, and out of focus; git history has them. Speech input, plan schemas, error-recovery loops, and a ROS2 VLM node are not planned.
- **Phase 4, scale and generalization:** VLA fine-tuning, parallel data generation, cross-simulator portability (for example Isaac Lab), a real SO-101 backend, and additional embodiments. These need compute and hardware beyond the current setup.
