# Project Roadmap: physai-robot-starter

`physai-robot-starter` is an open-source starter kit for embodied AI and robotics research. It provides stable robot, task, observation and action contracts so the same policy can be evaluated across transports (direct, ROS2, and in the future real robots) and simulator engines (MuJoCo, Isaac Sim).

## Supported robots and scope

| Robot | Status | Notes |
| --- | --- | --- |
| SO-101 | Supported, **current focus** | All new research work goes here |
| TurtleBot4 | Supported, maintained | Existing navigation code and tests stay green; no new work |
| Others | Planned | Added later through the robot adapter interface, only after SO-101 has a solid report |

- **Simulation only for now.** No hardware work is planned; the architecture stays open for it.
- **Out of scope for now:** transparent or thin real-world objects for SO-101; a TurtleBot4 research benchmark (after the SO-101 v0.2 report).

## Research story

Phase 2 asks one question: **why is learning needed for manipulation, and how far does it get us?** Three methods are compared with the same seeds, tasks and metrics, and the difficulty sweep (lighting, camera shift, clutter, distractors, occlusion) is the evidence:

| Method | Perception | Role |
| --- | --- | --- |
| Scripted expert | Privileged simulator state | Upper bound and demonstration teacher ([results](research/scripted_experts/README.md#results)) |
| Classical vision + state machine | Camera only | What hand-engineered perception and control can do ([findings](research/classical_control/FINDINGS.md)) |
| ACT (imitation learning) | Camera + proprioception | Does learning from pixels close the gap? ([findings](research/imitation_learning/FINDINGS.md)) |

## Status

Done, with focused tests: Phase 1 (capability-aware contracts, the SO-101 MuJoCo baseline and ROS2 bridge, FK/IK and safety validation, seeded domain randomization, the TurtleBot4 navigation baseline), the training bridge (observation/action specs, Gymnasium adapter, versioned datasets and checkpoints, shared evaluation reports), the classical vision baseline (100/100 nominal on seeds 0-99 on MuJoCo and Isaac, the difficulty sweep, missed-grasp detection with retry) and the Isaac Sim backend with sim-to-sim parity tiers 1 to 4. Numbers and causes are in the research FINDINGS; design is in [DECISIONS.md](docs/DECISIONS.md).

`[ ]` below marks planned, missing or partial work.

## Phase 2: benchmark, baselines and learning

### 2.0 Task ladder and capability report

| Level | Task | Tests |
| --- | --- | --- |
| T0 | Single cube, fixed setup | Scripted 300/300, done |
| T1 | Single cube, randomized pose / color / lighting / camera | Perception robustness |
| T2 | Cube size variation | Gripper aperture limits |
| T3 | Shape variation (cylinder, sphere, prism) | Grasp difficulty |
| T4 | Multi-object sorting: 3 colored cubes into 3 bins, then toward 9 | Multi-step planning and perception |
| T5 (stretch) | Clutter, distractors, occlusion, stacking | Hard perception and contact |

Sorting (3 cubes) already reaches 300/300 with the scripted expert. Evaluate on **at least 100 seeds** (20 cannot resolve reliability).

- [ ] Task definitions and scripted experts for T1-T4 (T5 stretch), each with fixed seeds.
- [ ] `scripts/capability_report.py`: reachable workspace, min/max graspable size, placement repeatability.
- [ ] Trajectory-quality metrics for the scripted expert (completion time, path length, jerk, joint-limit margin, peak speed) and a smoother variant such as minimum-jerk.

Done when: the scripted expert reaches about 100% on T0-T4 (lower confidence bound reported, or the limit documented), T0 and T1 are re-evaluated on at least 100 seeds, and the capability report says what sizes, shapes and positions the SO-101 can handle.

### 2A Classical vision baseline (required)

A camera-only state machine, all actions through the safety layer: the "before learning" reference ([README](research/classical_control/README.md), [FINDINGS](research/classical_control/FINDINGS.md)).

- [ ] A fair tuning effort so it is not a strawman: the clutter failures remain and need a grasp check, not a threshold.
- [ ] Recover from a missed grasp (retrying from the front or wrist estimate recovered 2 of 41 and 3 of 28 episodes; not tried: approach from the other side, change the pick offset, move the box).

Done when: evaluated on T0-T4 and the difficulty sweep with the shared protocol, failure modes documented.

### 2B Imitation learning with ACT

Train from scripted-expert demonstrations on a single 6 GB laptop GPU. One checkpoint (100 demos, 30k steps) already scores 92/100 on MuJoCo seeds 1000-1099 ([FINDINGS](research/imitation_learning/FINDINGS.md)).

- [ ] Fix and document the training configuration (image size, chunk size, batch size, precision).
- [ ] `collect_demos.py` exports the standard `LeRobotDataset` format (currently a LeRobot-shaped `.npz`).
- [ ] Reproducible training (fixed seeds, logged loss curves, checkpoint metadata) and at least two ablations (number of demonstrations, camera views, randomization on/off).

Done when: 50-100 demonstrations per task, reproducible training, evaluation on the shared protocol for T0-T4, categorized failure analysis.

### 2C Backend comparison study

Run the same checkpoint through direct MuJoCo and the ROS2 bridge to measure the integration layer, no hardware needed.

- [ ] ROS2-in-the-loop evaluation path in `eval_policy.py`.
- [ ] Per backend: success, end-to-end latency, effective control rate, action deviation, unsafe-action rejections; optional perturbations (latency, reduced rate, camera noise).

### 2E MuJoCo and Isaac Sim comparison

Local RTX GPU only, never installed by CI. Design: [DECISIONS.md D](docs/DECISIONS.md#d-isaac-sim-is-an-optional-peer-engine); measurements: [FINDINGS](research/classical_control/FINDINGS.md). `visual_servo` matches 100/100 on both engines; ACT scores 92/100 on MuJoCo and 54/100 on Isaac.

- [ ] Close the ACT sim-to-sim gap. The cameras differ strongly (mean absolute pixel difference 56 front, 47 wrist) while the cube position agrees, which is the leading suspect and is untested; next try colour and lighting augmentation in training.

### 2D Report and release (v0.2)

- [ ] 2-4 page report in `docs/`: setup, task ladder, three-method comparison, difficulty sweep, ablations, backend comparison, failure analysis, limitations.
- [ ] Tag `v0.2` with the exact configs, seeds and checkpoints to reproduce every reported number.

### Evaluation protocol (shared by all methods)

Held-out seeds: at least 50 per task and condition (at least 100 for T0/T1). Success rate with a 95% Wilson confidence interval. Metrics: success, collision, timeout, unsafe action, completion time. Difficulty axes: lighting, camera shift, clutter, distractors, occlusion. Optional after the report: state-based deep RL (PPO or SAC).

## Phase 3 and 4: not in focus yet

Listed only so the architecture keeps room for them. **Phase 3, language and planning:** only the planner contract and its scripted backend remain (`physai.planner`, `scripts/plan_task.py`); model-backed planners are not planned. **Phase 4, scale and generalization:** VLA fine-tuning, parallel data generation, a real SO-101 backend and additional embodiments.
