# Project Roadmap: physai-robot-starter

`physai-robot-starter` is an open-source starter kit for embodied AI and robotics research. It provides stable robot, task, observation and action contracts so the same policy can be evaluated across transports (direct, ROS2, and in the future real robots) and simulator engines (MuJoCo, Isaac Sim, Newton).

This roadmap lists framework features. Research studies built on the framework are listed in the private [`physai-research`](https://github.com/PhysAI-Robot/physai-research) repo (`research/README.md#studies`). `[x]` is shipped, `[ ]` is planned or partial. Design rationale is in [DECISIONS.md](docs/DECISIONS.md).

Simulation only for now; no hardware work is planned.

## Robots

- [x] SO-101 arm: MuJoCo, Isaac Sim and Newton backends, FK/IK, safety layer.
- [x] ViperX 300 S arm (a second manipulator, six joints): MuJoCo, Isaac Sim and Newton backends, FK/IK, grasp profile.
- [x] TurtleBot4: navigation baseline, maintained, no new work.
- [ ] Further embodiments through a `RobotDescriptor`.
- [ ] Real SO-101 backend (`ros2_real` is a contract-only placeholder today).

## Simulation engines

- [x] MuJoCo as the default engine.
- [x] Isaac Sim as an optional peer engine (local RTX GPU, never installed by CI), with sim-to-sim parity checks.
- [x] Newton as an optional third engine, and a renderer chosen apart from the physics (`--sim`, `--render`).
- [ ] Parallel episode generation for data collection and evaluation.

## Tasks and scenes

- [x] Single-cube pick-and-place (fixed target), three-cube sorting and a three-cube stack (`stack_cubes`, SO-101).
- [x] Seeded domain randomization (lighting, camera shift, clutter, distractors).
- [x] Spawn sampling for cube and place target over the reachable workspace (the region `scripts/workspace_map.py` reports), shared by both engines.
- [x] Minimum cube-to-target distance option.
- [x] `randomize_target` on Isaac Sim.
- [ ] Object variation: size, shape, then multi-object scenes.
- [ ] `scripts/capability_report.py`: min/max graspable size and placement repeatability (the reachable workspace is already `scripts/workspace_map.py`).

## Contracts, policies and planners

- [x] Capability-aware robot, task, observation and action contracts.
- [x] Policy, planner, scene, task and robot registries; plugins load by name from a manifest's `plugins:` without core importing them. A core `demo` policy drives the SO-101.
- [x] Planner contract with scripted backends (`ScriptedPlanner`, `StackPlanner`; `scripts/plan_task.py`) executed by `PlanRunner`, and a per-arm `GraspProfile` so one plan runs on SO-101 and ViperX 300 S.
- [ ] Model-backed planners and VLA policies, only through the existing `Planner` and `Policy` contracts.

## Data and training bridge

- [x] Observation/action specs, Gymnasium adapter, versioned datasets and checkpoints.
- [ ] Dataset collection exports the standard `LeRobotDataset` format (currently a LeRobot-shaped `.npz`).
- [ ] Reward and observation hooks sufficient for RL training through the Gymnasium adapter.

## Evaluation

- [x] Shared evaluation reports with Wilson 95% intervals, sharded runs, difficulty sweeps (`scripts/eval_policy.py`; sharded runs and sweeps are in the research plugin).
- [ ] Trajectory-quality metrics in the report: completion time, path length, jerk (done in `eval_policy.py`), joint-limit margin, peak speed.
- [x] Per-region success breakdown over the workspace (a plot script in the research plugin, from `cube_start` and `target_pos` in each result row).

## Transports and bridges

- [x] Direct and ROS2 simulation transports; ROS2 bridge.
- [ ] ROS2-in-the-loop evaluation path in `eval_policy.py`, with latency and control-rate measurement.

## Viewer and tooling

- [x] Browser viewer ([runbook](docs/WEB_VIEWER_RUNBOOK.md)), video recording, camera comparison across engines.
- [ ] Heatmap and per-seed failure browsing in the viewer.
