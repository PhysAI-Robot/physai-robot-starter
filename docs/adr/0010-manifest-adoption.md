# 10. Adopt the session manifest as the run description

## Status

Accepted. Extends [ADR 2](0002-session-manifest.md).

## Context

ADR 2 added `SessionManifest`, but nothing consumed it. `scripts/run_sim.py`
still had three input paths (`--config` task file, `--world` world file, a
bare `--robot`), each building its objects by hand and branching on robot
names, and the same defaults were typed in several places: the seed had a
three-level fallback, the pick-place success tolerance and hold length lived
in four files, and the TurtleBot4 path once crashed because a `None` step
limit reached its config.

## Decision

- Extend the schema additively, so `schema_version` stays 1: a `world` block,
  per-robot `model` and `config` (robot env fields), `success_hold_steps`, and
  normalisation of scene overrides and robot config (lists become tuples,
  `robot_xml` and `model` become paths). Several robots imply a shared world.
- Add `physai.runtime.create_session()`. A one-robot manifest becomes a
  `create_runtime()` composition; a `world` becomes a `SharedWorld`, whose
  robots run no task or policy yet.
- Convert the old inputs in `physai.config.compat` and build every `run_sim.py`
  run from the resulting manifest. `--config` and `--world` are deprecated and
  print a notice; they and their files are removed after a deprecation window.
- Give each default one source: the manifest's `simulation` block owns seed
  and domain randomization (a robot `config` that repeats them is rejected),
  the task and `TaskRuntime` own the success defaults, the robot registry owns
  a robot's default task, and a robot's capabilities decide the shape of the
  `constant` policy.
- A host-driven session (`Host`) steps the bare robot and composes no task,
  because a task would end episodes on success; the task only picks the scene.
  The host's camera thread renders only for robots that accepted
  `camera_stride=0`. TurtleBot4 renders inline, and a second GL context on
  another thread fails on Windows.

## Consequences

One representation describes every run, and a new robot needs no branch in
`run_sim.py`. `configs/manifests/` holds the examples, and tests keep the
shipped manifests equal to the legacy files they replace until those are
removed. Other scripts (`collect_demos.py`, `eval_policy.py`, and so on)
still assemble robots by hand, and shared-world sessions still cannot run a
task or policy; both are tracked in
[ARCHITECTURE.md](../ARCHITECTURE.md#known-remaining-gaps).
