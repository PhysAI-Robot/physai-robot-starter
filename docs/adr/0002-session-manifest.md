# 2. Introduce a single session manifest schema

## Status

Accepted

## Context

Configuration was split across three incompatible schemas:
`configs/sim_config.yaml` (global seed/randomization only),
`configs/tasks/<robot>/*.yaml` (single robot + task + scene + env, no policy
or backend field), and `configs/worlds/*.yaml` (multi-robot placement only,
no task or policy field). No file could express "N heterogeneous robot
instances + a task + a policy + a backend + viewer options" in one place,
and `run_sim.py` treated `--world` and `--config`/`--robot` as mutually
exclusive.

## Decision

Add a new, additive `SessionManifest` schema (`physai/config/manifest.py`,
loaded from `configs/manifests/*.yaml`) binding: `robots` (list of
`{id, robot, pose, task?, policy?}`), a scene, a `backend`
(`direct | ros2_sim | ros2_real`), and `viewer` options. A single-robot run
is simply a manifest with one entry in `robots`. The three existing loaders
(`load_sim_config`, `load_task_config`, `load_world_config`) stay for CLI
back-compat; nothing about today's flags changes.

## Consequences

New multi-robot, multi-backend sessions are expressible in one file.
Existing configs and CLI flags are unaffected. `ros2_real` is accepted by
the schema for forward compatibility but is not implemented, per
`ROADMAP.md`'s "simulation only for now."
