# Project Roadmap: physai-robot-starter

`physai-robot-starter` is an open-source starter kit for embodied AI and robotics research. It provides stable robot, task, observation and action contracts so the same policy can be evaluated across transports (direct, ROS2, and in the future real robots) and simulator engines (MuJoCo, Isaac Sim).

## Supported robots and scope

| Robot | Status | Notes |
| --- | --- | --- |
| SO-101 | Supported, **current focus** | All new research work goes here |
| TurtleBot4 | Supported, maintained | Existing navigation code and tests stay green; no new work |
| Others | Planned | Added later through the robot adapter interface, only after SO-101 has a solid report |

- **Simulation only for now.** No hardware work is planned; the architecture stays open for it.
- **Out of scope for now:** transparent or thin real-world objects for SO-101 (revisit after the v0.2 report); a TurtleBot4 research benchmark (task ladder, baselines, learned policies; after the SO-101 v0.2 report).

## Research story

Phase 2 answers one question: **why is learning needed for manipulation, and how far does it get us?** Three methods are compared with the same seeds, tasks and metrics:

| Method | Perception | Role |
| --- | --- | --- |
| Scripted expert | Privileged simulator state | Upper bound and demonstration teacher; 100% on single-cube and sorting (300 seeds each), see the [results table](research/scripted_experts/README.md#results) |
| Classical vision + state machine | Camera only | What hand-engineered perception and control can do without cheating |
| ACT (imitation learning) | Camera + proprioception | Does learning from pixels close the gap or extend beyond the classical limits? |

All methods are evaluated on a difficulty sweep (lighting, camera shift, clutter, distractors, occlusion). The expected result is that the classical pipeline degrades as perception gets harder; the curves are the evidence, not the narrative.

## Status

`[x]` = deliverable exists with focused test coverage. `[ ]` = planned, missing or partial.

```
Phase 1 (done) -> Bridge (done) -> Phase 2: benchmark -> classical baseline -> ACT -> backend study -> v0.2 report
                                                                                                          |
                                                                       Phase 3 and 4: not in focus yet
```

## Phase 1: Foundation and ROS2 contract (complete)

- [x] Capability-aware `Observation -> Action` contracts, robot registry, unit and frame validation, deterministic resets, seeded regression coverage.
- [x] SO-101 MuJoCo baseline: scripted single-cube pick-and-place, 300/300 on seeds 0-299 ([results](research/scripted_experts/README.md#results)).
- [x] SO-101 ROS2 bridge: joint, gripper, camera, TF, teleoperation, `rclpy` acceptance coverage.
- [x] SO-101 FK, Jacobian, numerical IK, Cartesian targeting (the ROS2 service endpoint is not bound yet), joint-limit and collision safety validation.
- [x] Seeded domain randomization (physics, visuals, cameras, clutter) with the deterministic baseline preserved.
- [x] TurtleBot4 navigation baseline and ROS2/Nav2 path (maintained only).
- [x] Phase 1 runs without Phase 2+ dependencies (LeRobot, VLA).

## Training bridge (complete)

- [x] Canonical `ObservationSpec` / `ActionSpec` schemas and robot-owned training contracts.
- [x] Gymnasium adapter routed through the existing safety gate.
- [x] Explicit SO-101 action layout shared by policies, recorder, replay, ROS2 conversion and datasets.
- [x] Versioned dataset metadata; checkpoint metadata with compatibility validation.
- [x] Shared evaluation reports (success, collision, timeout, unsafe action, reward, held-out seeds).

## Phase 2: Benchmark, baselines and learning (current focus)

### 2.0 Task ladder and capability report

One fixed-seed evaluation per level:

| Level | Task | Tests |
| --- | --- | --- |
| T0 | Single cube, fixed setup | Scripted 300/300, done |
| T1 | Single cube, randomized pose / color / lighting / camera | Perception robustness |
| T2 | Cube size variation | Gripper aperture limits |
| T3 | Shape variation (cylinder, sphere, prism) | Grasp difficulty |
| T4 | Multi-object sorting: 3 colored cubes into 3 bins, then toward 9 | Multi-step planning and perception |
| T5 (stretch) | Clutter, distractors, occlusion, stacking | Hard perception and contact |

The scripted expert was far weaker than its historical `20/20` suggested (45% and 60% on two 20-seed ranges), which is why the 100-seed rule below exists; causes are in [FINDINGS](research/scripted_experts/FINDINGS.md).

- [x] Diagnose the scripted-expert timeouts and restore a high single-cube success rate.
- [x] Reduce the sorting transfer-slip and neighbor-clutter failures (300/300 on seeds 0-299).
- [ ] Task definitions and scripted experts for T1-T4 (T5 stretch), each with fixed seeds.
- [ ] `scripts/capability_report.py`: reachable workspace, min/max graspable size, placement repeatability.
- [ ] Trajectory-quality metrics for the scripted expert: completion time, path length, jerk, joint-limit margin, peak speed.
- [ ] Compare the scripted trajectory with a smoother variant (for example minimum-jerk); smooth, consistent demonstrations matter for imitation learning.

Done when: the scripted expert reaches about 100% on T0-T4 (lower confidence bound reported, or the limit documented as a finding); T0 and T1 are re-evaluated on **at least 100 seeds**; and the capability report says what sizes, shapes and positions the SO-101 can handle in simulation.

### 2A Classical vision baseline (required)

A camera-only state-machine pipeline, all actions through the safety layer: the "before learning" reference. Results and failure analysis: [classical_control](research/classical_control/README.md) and its [FINDINGS](research/classical_control/FINDINGS.md).

- [x] Perception without simulator ground truth (`ColorBlobDetector`, calibration only, never object pose) and the approach, grasp, lift, place, release state machine (`SO101VisualServoPolicy`, the `visual_servo` policy).
- [x] All actions pass the safety layer: `SafetyController` gates the direct path inside `DirectAdapter`, not only the ROS2 and Gymnasium paths.
- [x] Nominal 100/100 on seeds 0-99 (MuJoCo and Isaac) and 50/50 on held-out seeds 100-149; the earlier timeouts were the static jaw landing on the cube.
- [x] Position error, settling time and categorized failure reasons: `scripts/report_evaluation.py` and `scripts/sweep_difficulty.py` (lighting, camera shift, clutter; 50 held-out seeds per cell on MuJoCo, 20 on Isaac). Lighting does not matter; an unmodelled camera shift does (20 mm: 23/50 MuJoCo, 3/20 Isaac) because of the wrist camera mount; clutter costs 3 to 6 episodes in 50.
- [x] Detect a missed grasp from the wrist camera and joint angles, retry twice, then stop with `grasp_missed` (nominal unchanged).
- [ ] A fair tuning effort so it is not a strawman: one time-boxed pass is done; the clutter failures remain and need a grasp check, not a threshold.
- [ ] Recover from a missed grasp. Retrying from the front or wrist estimate recovered 2 of 41 and 3 of 28 retried episodes, because the cause (a box under the jaw, a shifted wrist camera) is not a position error. Not tried: approach from the other side, change the pick offset after a miss, move the box.

Done when: evaluated on T0-T4 and the difficulty sweep with the shared protocol, with failure modes documented.

### 2B Imitation learning with ACT

Train from scripted-expert demonstrations on a single 6 GB laptop GPU.

- [ ] Fix and document the training configuration (image size, chunk size, batch size, precision).
- [ ] `scripts/collect_demos.py`: export the standard `LeRobotDataset` format (currently a LeRobot-shaped `.npz`).
- [ ] `research/imitation_learning/train_act.py`: fixed seeds, logged loss curves, checkpoint metadata.
- [ ] `scripts/eval_policy.py`: closed-loop held-out evaluation with success, collision, timeout and unsafe-action counts.
- [ ] At least two ablations (number of demonstrations, camera views, randomization on/off).

Done when: 50-100 demonstrations per task, reproducible training, evaluation on the shared protocol for T0-T4, categorized failure analysis.

### 2C Backend comparison study

Run the same checkpoint through direct MuJoCo and the ROS2 bridge to measure the effect of the integration layer, no hardware needed.

- [ ] ROS2-in-the-loop evaluation path in `scripts/eval_policy.py`.
- [ ] Per backend: success, end-to-end latency, effective control rate, action deviation, unsafe-action rejections.
- [ ] Optional controlled perturbations: added latency, reduced control rate, camera noise.

Done when: identical seeds on both backends, a results table, and the main causes of any gap identified.

### 2E MuJoCo and Isaac Sim sim-to-sim comparison

Evaluate a policy tuned in MuJoCo against Isaac Sim to measure the gap between simulators (2C measures integration overhead instead). Local RTX GPU only, never installed by CI. Design: [simulators ADRs](docs/adr/simulators.md); measurements and root causes: [FINDINGS](research/classical_control/FINDINGS.md).

- [x] Isaac backend (`physai.sim.isaac`, `robots.so101.isaac_env.SO101IsaacEnv`), verified end to end on Isaac Sim 6.1 (RTX 3060): URDF import, actuator gains, closed-loop joint tracking, camera rendering.
- [x] Parity tier 1-2 (FK and joint order, actuator step response) as `pytest.mark.isaac` tests.
- [x] Parity tier 3 (contact): a cube grasp-hold test passes on Isaac with under 1 mm drift over 12 s, matching MuJoCo's tolerance; PhysX friction-combine is forced to "max" to match MuJoCo.
- [x] Parity tier 4 (closed loop) for `visual_servo`: identical seeds, scene and perception on both engines (the same manifest runs on both with `--sim isaac`); matched result on seeds 0-99 at 320 x 240 is **100/100 on MuJoCo and Isaac** (all seeds agree; a second Isaac run also 100/100); 640 x 480 and 1280 x 720 are 20/20 and 8/8. Needed: matching pad contact geometry, a CLOSE-exit timing fix, a saturation/hue detector, linear render output, and no physics stepping in rendering.
- [x] `ArmKinematics` partly converted so Isaac can reuse its FK/IK math; full-scene collision queries (`forbidden_contact_body_pairs`) stay MuJoCo-only until something needs them.
- [x] `--sim isaac --serve` opens the web viewer with a display mirror of the arm, table, target and cube ([ADR 15](docs/adr/simulators.md#adr-15-simulator-engine-selection)).
- [x] ACT on Isaac Sim, in one environment with LeRobot ([ADR 18](docs/adr/simulators.md#adr-18-one-environment-for-isaac-sim-and-lerobot)). One checkpoint (100 demos, 30k steps) scores 92/100 on MuJoCo and 54/100 on Isaac over seeds 1000-1099; see [FINDINGS](research/imitation_learning/FINDINGS.md).
- [ ] Close the ACT sim-to-sim gap. The cameras differ strongly (background and table colour, mean absolute pixel difference 56 front, 47 wrist) while the cube position agrees, which is the leading suspect and is untested; next try colour and lighting augmentation in training.

Known limits: Isaac sometimes renders without the robot (detected by `robot_is_rendered()`; run long evaluations in shards and retry), PhysX is not deterministic run to run, and Isaac supports single-cube scenes with a fixed target, no randomization beyond a lighting scale and a camera jitter, and observation-only policies.

Done when: identical seeds and layouts across both simulators, a results table per tier, and the main causes of any gap identified.

### 2D Report and release (v0.2)

- [ ] 2-4 page report in `docs/`: setup, task ladder, three-method comparison, difficulty sweep, ablations, backend comparison, failure analysis, limitations.
- [ ] Tag `v0.2` with the exact configs, seeds and checkpoints to reproduce every reported number.

### Evaluation protocol (shared by all methods)

- Held-out seeds: at least 50 per task and condition (at least 100 for T0/T1).
- Success rate with a 95% confidence interval (for example Wilson).
- Metrics: success, collision, timeout, unsafe action, completion time.
- Difficulty axes: lighting, camera shift, clutter, distractors, occlusion.

Optional after the report: state-based deep RL (PPO or SAC) as a further comparison; vision-based RL only if the state-based version works.

## Phase 3 and Phase 4: not in focus yet

Listed only so the architecture keeps room for them.

- **Phase 3, language and planning:** only the planner *contract* and its scripted backend remain (`physai.planner`, `scripts/plan_task.py`). The SmolVLM and Claude backends were removed in the 2026-09-20 cleanup (untested, unrunnable without absent dependencies, out of focus; git history has them). Speech input, plan schemas, error-recovery loops and a ROS2 VLM node are not planned.
- **Phase 4, scale and generalization:** VLA fine-tuning, parallel data generation, a real SO-101 backend and additional embodiments. Cross-simulator portability moved to 2E once a GPU was available; the rest needs compute and hardware beyond the current setup.
