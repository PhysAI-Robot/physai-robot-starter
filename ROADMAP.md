# Project Roadmap: physai-robot-starter

`physai-robot-starter` is an open-source starter kit for embodied AI and robotics research. It provides stable robot, task, observation, and action contracts so the same policy can be evaluated across transports (direct, ROS2, and in the future real robots) and simulator engines (MuJoCo, Isaac Sim).

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
| Scripted expert | Privileged simulator state | Intended upper bound and demonstration teacher — 100% on single-cube and sorting (300 seeds each); see the [results table](research/scripted_experts/README.md#results) |
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
- [x] SO-101 MuJoCo baseline: scripted single-cube pick-and-place, 300/300
      on seeds 0-299 (see the [results table](research/scripted_experts/README.md#results)).
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
| T0 | Single cube, fixed setup | Scripted 300/300 — done |
| T1 | Single cube, randomized pose / color / lighting / camera | Perception robustness |
| T2 | Cube size variation | Gripper aperture limits |
| T3 | Shape variation (cylinder, sphere, prism) | Grasp difficulty |
| T4 | Multi-object sorting: 3 colored cubes into 3 bins, then scale toward 9 | Multi-step planning and perception |
| T5 (stretch) | Clutter, distractors, occlusion, stacking | Hard perception and contact |

**Resolved finding (2026-09-21).** The scripted expert was far weaker than the
historical `20/20` suggested (a 20-seed run scored 45% and 60% on two seed
ranges), which is why the ≥100-seed rule below exists. Three bugs were fixed;
the root causes and reverted experiments are in
[research/scripted_experts/FINDINGS.md](research/scripted_experts/FINDINGS.md)
and current rates are in the
[results table](research/scripted_experts/README.md#results).

Deliverables:

- [x] Diagnose the scripted-expert timeouts and restore a high single-cube success rate.
- [x] Diagnose and reduce the sorting transfer-slip and neighbor-clutter failures (now 300/300 on seeds 0-299).
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

- [x] Perception module without simulator ground truth (`ColorBlobDetector` in `research/classical_control/so101_visual_servo.py`; reads camera calibration only, never object pose).
- [x] State machine covering approach, grasp, lift, place, and recovery on failure (`SO101VisualServoPolicy`, registered as the `visual_servo` policy).
- [x] All actions pass the safety layer: `SafetyController` now gates the direct path inside `DirectAdapter`, not only the ROS2 and Gymnasium paths.
- [x] Diagnose the `visual_servo` timeouts (95/100 on seeds 0-99 after the fingertip pad refit). They were seeds where the static jaw landed on the cube; hovering until the pinch is aligned, a 6 mm safe-side pick offset and `fx = fy` in the camera calibration fix them: 100/100 on seeds 0-99 and the 20-seed CI check passes at every resolution (research/classical_control/FINDINGS.md).
- [ ] Report position error, settling time, and categorized failure reasons.
- [ ] Give it a fair tuning effort; it must not be a strawman.

Definition of done: evaluated on T0-T4 and the difficulty sweep with the shared protocol, with its failure modes documented.

### 2B Imitation learning with ACT

Train from scripted-expert demonstrations. The configuration must fit a single 6 GB laptop GPU.

- [ ] Fix and document the training configuration (image size, chunk size, batch size, precision).
- [ ] `scripts/collect_demos.py`: export standard `LeRobotDataset` format (currently a LeRobot-shaped `.npz`, partial).
- [ ] `research/imitation_learning/train_act.py`: fixed seeds, logged loss curves, checkpoint metadata.
- [ ] `scripts/eval_policy.py`: closed-loop held-out evaluation with success, collision, timeout, and unsafe-action counts.
- [ ] At least two ablations (for example number of demonstrations, camera views, randomization on/off).

Definition of done: 50-100 demonstrations per task, reproducible training, evaluation on the shared protocol for T0-T4, and categorized failure analysis.

### 2C Backend comparison study

Run the same checkpoint through direct MuJoCo and through the ROS2 bridge to measure the effect of the integration layer, with no hardware needed.

- [ ] ROS2-in-the-loop evaluation path in `scripts/eval_policy.py`.
- [ ] Metrics per backend: success, end-to-end latency, effective control rate, action deviation, unsafe-action rejections.
- [ ] Optional controlled perturbations: added latency, reduced control rate, camera noise.

Definition of done: identical seeds on both backends, a results table, and the main causes of any gap identified.

### 2E MuJoCo-Isaac Sim sim-to-sim comparison

Evaluate a policy tuned in MuJoCo (`visual_servo`, ACT) against Isaac Sim to
measure the gap between simulators, not simulation-vs-ROS2 integration
overhead (that is 2C). Local RTX GPU only; never installed by CI (see
`docs/adr/0016-isaacsim-as-a-project-extra.md` for the `--extra isaac`
install, `docs/adr/0013-isaac-sim-optional-backend.md` for the backend
itself).

- [x] `physai.sim.isaac` (SimulationApp lifecycle, `RobotDescription` -> USD) and
  `robots.so101.isaac_env.SO101IsaacEnv`, verified end to end against real
  Isaac Sim 6.1 on an RTX 3060 (URDF import, actuator gain transfer with no
  unit conversion, closed-loop joint tracking, camera rendering).
- [x] Parity ladder tiers 1-2 (static FK/joint-order match, actuator step
  response) as `pytest.mark.isaac` tests.
- [x] Parity ladder tier 3 (contact): a cube grasp-hold test
  (`tests/research/scripted_experts/test_so101_grasp_hold_isaac.py`),
  passing against real Isaac Sim on an RTX 3060 (<1 mm drift over 12 s,
  matching MuJoCo's own tolerance). `SO101IsaacEnv` grew a minimal,
  opt-in `cube` (`GraspCubeConfig`, `sim.isaac.objects.add_cube`) — not a
  `ManipulationSceneConfig` port (no table, target, layout, or
  randomization), scoped to exactly this test. `apply_contact_friction`
  was wired in (previously defined but never called) and both it and the
  cube's material force PhysX's friction-combine mode to "max", matching
  MuJoCo's own combine policy instead of PhysX's default (average).
- [x] Parity ladder tier 4 (closed-loop) for `visual_servo`: identical seeds
  and scene in both simulators, success rate and failure causes via
  `scripts/eval_policy.py --sim isaac` and `scripts/compare_evaluations.py`
  (ACT on Isaac is the next item). History of how it got there, starting
  with a first pass verified against real
  Isaac Sim (`tests/research/classical_control/test_so101_visual_servo_isaac.py`):
  the unmodified `SO101VisualServoPolicy` runs its full detect -> approach ->
  descend -> close -> lift -> transfer -> lower -> release -> retreat phase
  sequence end to end, and the front-camera vision/triangulation pipeline is
  accurate to ~1-2mm (fixed two real bugs along the way: USD's default
  camera `clippingRange` of `(1, 1e6)` stage units was clipping the entire
  workspace at this project's meter scale, in both `apply_cameras` and
  `add_world_camera`; and `camera_calibration()`'s `fx = fy * width /
  height` was wrong for a pinhole camera whose horizontal aperture is
  itself scaled by that same ratio — should just be `fx = fy`, since fixed
  in `mujoco_env.py` too, see below). The shared
  `SO101VisualServoPolicy` also takes an optional `squeeze_grip` override; it
  was needed on Isaac only while the pad geometry mismatched (see below) and
  the shared default `0.15` now works unmodified.

  **CLOSE-exit timing bug: fixed and verified (2026-09-29).** The original
  cause (see `research/classical_control/FINDINGS.md` for the full trace
  evidence): `act()`'s CLOSE/RELEASE settle check compared the actual
  gripper position against the *ramping* (fixed `0.9 rad/s`) commanded
  position, not the final target. MuJoCo's own gripper lags that ramp
  enough that this accidentally worked; Isaac's PhysX position drive tracks
  it with ~no lag, so CLOSE always exited after a fixed ~8 steps at
  whatever fraction of the ramp had elapsed by then — as little as ~21%
  closed against a `squeeze_grip=0.06` target, well before the squeeze
  reached the cube's actual jam point, with the arm already moving into
  LIFT while the grip was still mid-ramp. Fixed by requiring the ramp to
  finish (`ramp_done`) before accepting either `tracking` (actual position
  close to the now-fixed command) or a long-enough `stalled` (jaw stopped
  moving before reaching the command — the normal case once a squeeze is
  deep enough to jam) as settled; `_GRIP_STALL_SETTLE_STEPS` gives a
  transient stick-slip pause time to resolve into further closing on its
  own before it's accepted, which is what kept an earlier, shorter-patience
  version of this fix from also fixing a secondary symptom (the delayed
  slip-through landing mid-LIFT instead of mid-CLOSE, letting momentum
  fling the cube). Verified: 100/100 MuJoCo seeds unaffected (identical
  95/100 success, identical 5 timeout seeds, before and after), full test
  suite unaffected, and against real Isaac Sim CLOSE now reaches close to
  the genuine squeeze depth and the cube rises smoothly (1-4cm) instead of
  ~0 or a multi-cm single-step launch.

  **TRANSFER hold: fixed by matching contact geometry (2026-09-29).** The
  cube slipping out during TRANSFER was not a friction, force-cap, or solver
  problem: MuJoCo grips with two fitted 12 x 12 x 6 mm pad boxes, while Isaac
  had only ever received the pads' friction material on the raw jaw mesh.
  `sim.isaac.description.apply_contact_pad_colliders` now builds the same pads
  from the shared `contact_pads` spec (pad-centre distance at the alignment
  angle checked: 33.9 mm vs the expected 34 mm). With that, `visual_servo` on
  Isaac delivers the cube 9 mm from the target with the *shared default*
  `squeeze_grip=0.15` (no per-simulator override; only `target_plane_z`, which
  is scene geometry), passing 3 of 3 consecutive runs of
  `test_visual_servo_runs_the_full_pick_and_place_loop`, which now asserts
  delivery within 4 cm; tier 3 and the Isaac env tests still pass. Five
  parameter experiments run *before* this (a deeper `squeeze_grip`, raising
  `gripper_force_limit`, per-body and scene-wide PhysX solver iterations,
  doubling pad friction, compliant pad contact) each failed or regressed
  tier 3 — kept in `research/classical_control/FINDINGS.md` as the record of
  what not to try against mismatched geometry.

  **Identical scene, seeds and perception (2026-10-02/03).** Isaac builds the
  same workspace as MuJoCo from the shared scene config (table, target pad,
  cube, front camera), draws the cube per seed with the same RNG order
  (checked against `golden_layouts.json`), has a wrist-camera calibration,
  restores the cube on `reset()`, and a `TaskRuntime` reports success. Camera
  resolution is one of 320 x 240 / 640 x 480 / 1280 x 720 for every
  simulator, script and test (default 320 x 240; `--camera-res` or the
  manifest's `simulation.camera_resolution`), with `fx = fy` in both. The
  policy has new shared defaults (`align_before_descend`, `grasp_offset_xy`).
  The same manifest runs on both engines (`run_sim.py --manifest
  configs/manifests/so101_single_cube_fixed_place.yaml --sim isaac`). Matched result on
  seeds 0-99 at 320 x 240: **MuJoCo 100/100 and Isaac 100/100** (Wilson 95%
  96-100% each, all 100 seeds agree); 640 x 480 and 1280 x 720 are 20/20 in
  MuJoCo and 8/8 in Isaac. Getting there needed two perception fixes beyond
  geometry (Isaac scored 79/100 before them): a lighting-independent
  saturation/hue detector, and linear, gamma-free output so Isaac's cube is as
  saturated as MuJoCo's; and two measurement fixes in the Isaac env itself
  (rendering no longer advances physics, which had doubled simulated time per
  control step, and `reset()` holds one step so the first image is not stale).
  Arm motion now matches MuJoCo's to within a step per phase; Isaac's episodes
  are shorter (117 against 200 steps) only because its gripper settles in
  CLOSE sooner. Root causes, the glitch below and the measurements are in
  `research/classical_control/FINDINGS.md`. Known limits: Isaac renders
  without the robot in some processes (detected by
  `SO101IsaacEnv.robot_is_rendered()`; run long evaluations in shards and
  retry), PhysX is not deterministic run to run, and Isaac supports
  single-cube scenes with a fixed target, no domain randomization and only
  observation-based policies. `--sim isaac --serve` opens the web viewer with
  a display mirror of the arm (ADR 15).
- [ ] ACT on Isaac Sim. Blocked by a dependency conflict, not by the env:
  `[tool.uv] conflicts` forbids installing extra `vla` (torch + lerobot,
  `numpy<2.3`) together with `isaac` (`isaacsim` pins `numpy==2.3.1`), so ACT
  cannot run in the same process as Isaac. Options: an out-of-process policy
  server (Isaac sends observations, a `vla` venv returns actions), or two
  environments with recorded rollouts; either is its own piece of work.
- [x] `ArmKinematics` (`robots/so101/kinematics.py`) partially converted:
  `qpos_to_site_pose`/`pinch_center_from_qpos`, new additive methods aside
  `fk`/`tool_pose`/`pinch_center`'s existing `MjData`-taking ones, cover
  what tier 3 needed. `ik`/`ik_pinch` already took only joint positions
  and needed no change. `forbidden_contact_body_pairs` (full-scene
  collision detection, not just this arm's FK) remains MuJoCo-only — no
  Isaac contact-query API is wired up yet; deferred until something
  besides `so101_pick_place_expert.py`'s own MuJoCo-only `_cube_grasped`
  needs it.

Definition of done: identical seeds/layouts across both simulators, a
results table per tier, and the main causes of any gap identified.

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
- **Phase 4, scale and generalization:** VLA fine-tuning, parallel data generation, a real SO-101 backend, and additional embodiments. Cross-simulator portability moved to 2E (Isaac Sim) once a GPU became available; these remaining items still need compute and hardware beyond the current setup.
