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
| Scripted expert | Privileged simulator state | Intended upper bound and demonstration teacher — currently underperforms the classical baseline, see Phase 2.0 finding |
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
      2026-09-20 at **9/20 over 20 seeds** (11 timeouts), not the 20/20
      recorded earlier; see the Phase 2.0 finding below.
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
| T0 | Single cube, fixed setup | Scripted 9/20 measured — needs work |
| T1 | Single cube, randomized pose / color / lighting / camera | Perception robustness |
| T2 | Cube size variation | Gripper aperture limits |
| T3 | Shape variation (cylinder, sphere, prism) | Grasp difficulty |
| T4 | Multi-object sorting: 3 colored cubes into 3 bins, then scale toward 9 | Multi-step planning and perception |
| T5 (stretch) | Clutter, distractors, occlusion, stacking | Hard perception and contact |

**Open finding (2026-09-20).** The scripted expert is weaker than previously
recorded. Measured with the documented commands over 20 seeds:

| Policy | Variant | Success | Failure mode |
| --- | --- | --- | --- |
| `scripted` | Single cube | 9/20 (45%) | 11 timeouts |
| `scripted` | Sorting | 1/20 (5%) | 19 timeouts |
| `visual_servo` | Single cube | **20/20 (100%)** | none |

Timeouts, not collisions or drops, dominate the scripted failures.

This inverts an assumption in the research story above. The camera-only
`visual_servo` baseline outperforms the scripted expert that reads cube pose
straight from the simulator, so the scripted expert is **not** currently an
upper bound and is a poor demonstration teacher. Either the scripted expert is
repaired before 2B collects demonstrations, or `visual_servo` becomes the
teacher and the three-method comparison is restated.

Deliverables:

- [ ] Diagnose the scripted-expert timeouts and restore a high single-cube success rate.
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
