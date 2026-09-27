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
  isaac` combined with `--viewer`/`--serve` before attempting either (both
  are MuJoCo-only — see Consequences), and its module-level `import mujoco`
  moved into the `--viewer` branch so a headless `--sim isaac` run never
  needs MuJoCo importable.
- `SO101Env.robot_spec.metadata` gains `"simulator": "mujoco"`, matching
  `SO101IsaacEnv`'s existing `"simulator": "isaac"` — a recorded dataset now
  says which engine produced it either way.

`scripts/collect_demos.py` and `scripts/eval_policy.py` do **not** gain
`--sim`: neither calls `create_runtime`/`create_session` at all — both
hand-build a MuJoCo `EnvConfig` directly and, for `collect_demos.py`, use
`SO101PickPlaceExpert` (a privileged ground-truth expert reading MuJoCo
object pose and `ArmKinematics` directly) and raw `robot.data.qpos` access.
Adding the flag would not work today regardless: `IsaacEnvConfig` has no
scene/task objects and no matching expert. This is the existing "scripts/
composition-root adoption" gap in `docs/ARCHITECTURE.md`'s known-gaps list,
not something to paper over with a flag that always errors.

## Consequences

A manifest or `--sim isaac` run now fails at load time with a specific
message (unsupported robot, conflicting `config.simulator`, or an
incompatible `world`/`backend`/`viewer` combination) instead of failing deep
inside a robot factory or, worse, silently misbehaving (the `_robot_fields`
bug). `configs/manifests/so101_isaac.yaml` is the worked example: no `task`
(required, since `SO101IsaacEnv` has no scene/task objects yet), headless.

`scripts/run_sim.py --viewer`/`--serve`, `physai.web` more generally,
`physai.sim.mujoco.world.SharedWorld`, the ROS2 bridge, and
`robots/so101/kinematics.py`'s `ArmKinematics` all remain MuJoCo-only; this
ADR makes that explicit at the manifest-validation boundary but does not
remove any of those limitations, which are tracked in
`docs/ARCHITECTURE.md`'s "Known remaining gaps".
