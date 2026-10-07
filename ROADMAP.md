# Project Roadmap: physai-robot-starter

`physai-robot-starter` is an open-source starter kit for embodied AI and robotics research. It provides stable robot, task, observation and action contracts so the same policy can be evaluated across transports (direct, ROS2, and in the future real robots) and simulator engines (MuJoCo, Isaac Sim).

This roadmap lists framework features. Research studies built on the framework are listed in [research/README.md](research/README.md#studies). `[x]` is shipped, `[ ]` is planned or partial. Design rationale is in [DECISIONS.md](docs/DECISIONS.md).

Simulation only for now; no hardware work is planned.

## Robots

- [x] SO-101 arm: MuJoCo and Isaac Sim backends, FK/IK, safety layer.
- [x] TurtleBot4: navigation baseline, maintained, no new work.
- [ ] Further embodiments through a `RobotDescriptor`, once the SO-101 feature set is complete.
- [ ] Real SO-101 backend (`ros2_real` is a contract-only placeholder today).

## Simulation engines

- [x] MuJoCo as the default engine.
- [x] Isaac Sim as an optional peer engine (local RTX GPU, never installed by CI), with sim-to-sim parity checks.
- [ ] Parallel episode generation for data collection and evaluation.

## Tasks and scenes

- [x] Single-cube pick-and-place (fixed target) and three-cube sorting.
- [x] Seeded domain randomization (lighting, camera shift, clutter, distractors).
- [ ] Spawn sampling for cube and place target over the reachable workspace (the region `scripts/workspace_map.py` reports), shared by both engines.
- [ ] Minimum cube-to-target distance option.
- [ ] `randomize_target` on Isaac Sim (`SO101IsaacEnv` rejects it today).
- [ ] Object variation: size, shape, then multi-object scenes.
- [ ] `scripts/capability_report.py`: reachable workspace, min/max graspable size, placement repeatability.

## Contracts, policies and planners

- [x] Capability-aware robot, task, observation and action contracts.
- [x] Policy, planner and robot registries; research code plugs in without core importing it.
- [x] Planner contract with a scripted backend (`physai.planner`, `scripts/plan_task.py`).
- [ ] Model-backed planners and VLA policies, only through the existing `Planner` and `Policy` contracts.

## Data and training bridge

- [x] Observation/action specs, Gymnasium adapter, versioned datasets and checkpoints.
- [ ] `collect_demos.py` exports the standard `LeRobotDataset` format (currently a LeRobot-shaped `.npz`).
- [ ] Reward and observation hooks sufficient for RL training through the Gymnasium adapter.

## Evaluation

- [x] Shared evaluation reports with Wilson 95% intervals, sharded runs, difficulty sweeps (`scripts/eval_policy.py`, `run_sharded_eval.py`, `sweep_difficulty.py`).
- [ ] Trajectory-quality metrics in the report: completion time, path length, jerk, joint-limit margin, peak speed.
- [ ] Per-region success breakdown over the workspace.

## Transports and bridges

- [x] Direct and ROS2 simulation transports; ROS2 bridge.
- [ ] ROS2-in-the-loop evaluation path in `eval_policy.py`, with latency and control-rate measurement.

## Viewer and tooling

- [x] Browser viewer ([runbook](docs/WEB_VIEWER_RUNBOOK.md)), video recording, camera comparison across engines.
- [ ] Heatmap and per-seed failure browsing in the viewer.
