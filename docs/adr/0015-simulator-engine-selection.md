# 15. Simulator engine selection as a manifest and registry concept

## Status

Accepted

## Context

[ADR 13](0013-isaac-sim-optional-backend.md) added Isaac Sim as a second
engine reachable only through `so101.factory.make_so101`'s own `simulator=`
kwarg, itself only reachable through a manifest robot's free-form `config:`
map (`config: {simulator: isaac}`). Nothing validated that a given robot
supported the requested engine before construction — TurtleBot4 has no Isaac
path, and nothing would have said so before failing deep inside its factory.
`runtime.session._robot_fields` filtered a robot's `config` keys against
MuJoCo's `EnvConfig` fields unconditionally, even when the robot would
actually be built with `IsaacEnvConfig` — a latent bug that happened not to
matter yet because no manifest had exercised the isaac path end to end.
There was also no command-line way to pick an engine; only a Python call or
a hand-written manifest could.

## Decision

Promote the simulator engine to the same kind of concept the robot registry
already has for transport adapters:

- `RobotDescriptor` gains a `simulators: tuple[str, ...] = ("mujoco",)`
  field. `so101` registers `("mujoco", "isaac")`; every other robot keeps
  the default.
- `robots.registry.create_robot(name, *, simulator="mujoco", ...)` validates
  `simulator` against the robot's `descriptor.simulators` before calling its
  factory, the same way it already validates the robot name itself.
  `available_simulators(name)` exposes the check for manifest validation.
- `create_env_config(name, *, simulator=None, ...)` forwards `simulator` to
  a robot's env-config factory only when that robot declares more than one
  simulator (`len(descriptor.simulators) > 1`) — a single-simulator robot's
  factory is never asked to accept or ignore a parameter it has no second
  value for. `so101_env_config()` (in `so101/factory.py`) is the new
  registered `env_config` factory for so101, replacing the bare `EnvConfig`
  class; it picks `EnvConfig` or `IsaacEnvConfig` the same way
  `make_so101()` picks `SO101Env` or `SO101IsaacEnv`.
- `runtime.create_runtime` gains a `simulator: str = "mujoco"` parameter,
  threaded straight through to `create_robot`.
- The session manifest gains an optional top-level `simulator: mujoco`
  field, next to `backend` (transport and engine are independent axes, so
  they are independent fields). Validated at load time: every robot in the
  session must support it (`available_simulators`); a robot's own `config`
  must not repeat it (same pattern as the existing `simulation`-block
  ownership check); and a non-`"mujoco"` engine rejects a `world` block (or
  more than one robot, which implies one), `backend: ros2_sim`, and any
  `viewer.mode` other than `none` — none of those paths support a second
  engine yet. This closes the `_robot_fields` bug above: it now asks
  `create_env_config` for the manifest's actual simulator, not an implicit
  MuJoCo default.
- `scripts/_common_args.add_simulator()` adds `--sim {mujoco,isaac}` to
  `scripts/run_sim.py`, applied through `config.compat.with_overrides`. A
  bare `dataclasses.replace()` (how every other override in `with_overrides`
  works) would skip the registry and cross-field checks above, so
  `with_overrides` re-runs them through the manifest module's new
  `validate_manifest()` (the same function `parse_manifest` itself calls)
  whenever `simulator` is overridden. `run_sim.py` also rejects `--sim
  isaac` combined with `--viewer` (MuJoCo's native window), or `--serve` with
  `--record-dir`, before attempting either; plain `--serve` is supported
  (see "Isaac `--serve`" below). Its module-level `import mujoco` moved into
  the `--viewer` branch so a headless `--sim isaac` run never needs MuJoCo
  importable.
- **Isaac `--serve`.** `run_sim.py --sim isaac --serve` reuses the web
  `Host`, with two Isaac-specific adaptations. Isaac must stay on the thread
  that created its `SimulationApp`, so the host loop runs on the main thread
  (`Host.run()`) and uvicorn runs on a worker thread with its own event loop
  (Kit replaces `asyncio.run` with a version that rejects uvicorn's
  `loop_factory`). `Host` and the browser read a MuJoCo `model`/`data`, so
  `SO101IsaacEnv` exposes a display/telemetry mirror: the kinematics-oracle
  model plus an `MjData` that `observe()` refreshes from Isaac's joint state
  and never simulates. The 3D view therefore shows the arm only (no task
  objects, no contacts); the cube appears in the Isaac-rendered camera feeds.
  Cartesian jog is absent (`so101_jog_resolver` returns `None` for an env
  without `resolve_twist_jog`) and recording/playback stay MuJoCo-only.
- `SO101Env.robot_spec.metadata` gains `"simulator": "mujoco"`, matching
  `SO101IsaacEnv`'s existing `"simulator": "isaac"` — a recorded dataset now
  says which engine produced it either way.

- **The same scene on both engines.** `IsaacEnvConfig.scene` takes the
  `ManipulationSceneConfig` MuJoCo builds from, so the table, cube, target,
  front camera and the per-seed cube position come from one source (the seed
  draws go through `robots.so101.layout.draw_cube_xy`, the order pinned by
  `golden_layouts.json`). `SO101IsaacEnv` also exposes `table_top`/
  `cube_half`, so a `TaskRuntime` wraps it like MuJoCo's env and reports
  `success`/`dist_cube_target`. One manifest therefore describes the
  environment for both engines: `run_sim.py --manifest
  configs/manifests/so101_single_cube_fixed_place.yaml --sim isaac`. Limits: single-cube
  scenes, a fixed target, no domain randomization beyond a lighting scale and
  a camera position jitter, and policies that read only observations
  (`visual_servo`, `constant`).
- `scripts/eval_policy.py` gains `--sim isaac` on that basis (same scene,
  seeds, task and report schema; `scripts/compare_evaluations.py` tabulates
  the two). `scripts/collect_demos.py` still does not: its
  `SO101PickPlaceExpert` is a privileged ground-truth expert reading MuJoCo
  object pose and `ArmKinematics` directly, which has no Isaac equivalent.

## Consequences

A manifest or `--sim isaac` run now fails at load time with a specific
message (unsupported robot, conflicting `config.simulator`, or an
incompatible `world`/`backend`/`viewer` combination) instead of failing deep
inside a robot factory or, worse, silently misbehaving (the `_robot_fields`
bug). `configs/manifests/so101_isaac.yaml` is the robot-only smoke
manifest (no scene, `policy: constant`); `so101_single_cube_fixed_place.yaml` with
`--sim isaac --policy visual_servo --video` runs the full task.

`scripts/run_sim.py --viewer`, `physai.web` recording/playback,
`physai.sim.mujoco.world.SharedWorld`, the ROS2 bridge, and
`robots/so101/kinematics.py`'s `ArmKinematics` all remain MuJoCo-only; this
ADR makes that explicit at the manifest-validation boundary but does not
remove any of those limitations, which are tracked in
`docs/ARCHITECTURE.md`'s "Known remaining gaps".
