# Architecture

The internal design reference: design shape, module ownership, dependency rules, stable
contracts, the session manifest, the host API, extension seams and the research/core
boundary. Setup and commands are in [README.md](../README.md), agent workflow rules in
[AGENTS.md](../AGENTS.md), and the reasons behind the frozen shape in
[DECISIONS.md](DECISIONS.md).

## Design shape

The stack separates embodiment, task and decision-making so each can evolve independently.
A runtime composition selects one or more robots, one task where applicable, and one planner
or policy per session, then connects them through message-shaped contracts:

```text
instruction + camera images -> Planner -> Plan (sub-goals, optional waypoints)
        -> Policy or PlanRunner at the control rate -> Action chosen by robot capabilities
        -> robot environment and task evaluation
```

The runnable baselines are the scripted planner/policy path with `PlanRunner` and the SO-101
visual-servo policy; model-backed planners and VLA policies plug into the same boundaries
from `research/`.

## Execution modes

A simulator simulates; ROS2 transports. They are independent axes:

```text
Fast development:        Planner/Policy -> direct RobotPort -> simulator
Integration validation:  Planner/Policy node -> ROS2 topics -> ROS2SimAdapter -> simulator
Real deployment:         Planner/Policy node -> ROS2 topics -> ROS2HardwareAdapter -> hardware
```

Both ROS2 paths expose the same topic, message, unit, frame, joint-order and rate contracts,
so a policy node moves from simulation to hardware unchanged. Direct mode is the fast path
for tests, training and regression; it does not replace ROS2 validation. `RobotPort` has
three transport adapters named after the manifest's `backend` value: `DirectAdapter`
(`direct`) and `ROS2SimAdapter` (`ros2_sim`) are implemented, `ROS2HardwareAdapter`
(`ros2_real`) is a contract-only placeholder (simulation only for now, see
[ROADMAP.md](../ROADMAP.md)). Each wraps whichever `RobotPort` a robot factory built, and
transports are registry-driven (`robots.adapters.register_adapter()`).

## Module ownership

| Module | Owns | Must not own |
| --- | --- | --- |
| `physai.contracts` | Shared `Observation`, `Action` and ROS2-shaped value types | Robot-specific ordering or task rules |
| `physai.config` | Typed YAML parsing: `manifest` (session manifest, `simulation` block) and `compat` (CLI overrides, `with_overrides`) | Simulation behavior, robot construction or task evaluation |
| `physai.robots` | Embodiment discovery, `RobotSpec`, ports, factories, environments, adapters, the backend registry, shared-world factories, the sim-neutral `RobotDescription` | Task reward, planner decisions or model SDKs |
| `physai.tasks` | Task state, reset rules, reward, metrics, termination | Robot internals or action generation |
| `physai.sim` | One subpackage per engine plus the neutral modules both read (`studio`: the common look; `workspace`: table, target, cubes, front camera) | Engine code in the namespace; importing `physai.sim` must never need an engine SDK |
| `physai.sim.mujoco` | MuJoCo core, scene builders, the shared multi-robot world, rendering, simulation time | Robot-specific environment logic, ROS2 transport, any robot-name branch |
| `physai.sim.isaac` | `SimulationApp` lifecycle, `RobotDescription` to USD, world extras, `add_workspace` | Robot-specific environment logic or task composition; importing `physai.sim.mujoco` |
| `physai.planner` | Instruction and image grounding, `Plan`, `SubGoal`, the planner registry | Control-rate motor commands; research planners |
| `physai.policy` | Control-rate `Action` production, core baselines (`constant`, `constant_twist`, `replay`), the policy registry | Task scoring, robot discovery, checkpoint-backed inference |
| `physai.control` | Action resolution, capability checks, rate limiting | High-level planning or task semantics |
| `physai.data` | Episode recording, dataset and checkpoint metadata, evaluation, the Gymnasium adapter | Simulation decisions or model inference |
| `physai.bridge` | ROS2 transport, topic and message mapping, timing, ROS2-backed adapters | Physics, task semantics or model inference |
| `physai.runtime` | Composition (`create_runtime`, `create_session`), compatibility checks, the episode rollout (`run_episode`) | Robot-specific physics, task reward or model inference |
| `physai.web` | The one `Host`, the FastAPI client surface, telemetry, the static browser client | Simulator selection or robot construction |
| `research/<topic>/` | Approach-specific implementations that self-register on import | Anything a core module imports |
| `scripts/` | CLI argument parsing and runtime composition | IK, reward calculation or SDK-specific code |
| `launch/` | Generic ROS2 launch composition | Robot physics, transport callbacks or task rules |

## Dependency direction

Declared in `pyproject.toml`'s `[tool.importlinter]`, checked by `uv run lint-imports` and by
`tests/core/boundaries/test_import_boundaries.py` (so `pytest tests/ -q` fails on a violation):

| Rule | Rationale |
| --- | --- |
| `physai.sim` must not import `physai.bridge` or `rclpy` | ROS2 stays optional for simulator-only workflows |
| `physai.policy`, `tasks`, `planner` must not import `mujoco` | They depend on ports. One accepted indirect path: `policy.registry` calls `robots.registry.create_robot_policy()` |
| `physai` must not import `research` | Research registers itself into a core registry on import |
| `physai.robots.description` and `physai.sim.workspace` must not import `mujoco` or Isaac | Every engine's builder reads them ([DECISIONS.md C](DECISIONS.md#c-sim-neutral-data)) |
| Only `physai.sim.isaac` and `physai.robots.so101.isaac_env` may import `isaacsim`, `omni`, `pxr` | Isaac is optional and GPU-only ([D](DECISIONS.md#d-isaac-sim-is-an-optional-peer-engine)) |
| `physai.sim.mujoco` and `physai.sim.isaac` must not import each other; Isaac must not import `mujoco` | Peer engines |
| A robot-name branch in generic code is a bug | `SharedWorld`, `Host` and the web client resolve behavior through `RobotSpec` capabilities or a registry. Enforced by review, not import-linter |

## Design patterns

- **Ports and adapters.** `Observation`, `Action`, `RobotPort`, `Planner`, `Policy` and `Task`
  are ports; simulators, ROS2 transport, model SDKs and concrete robots are adapters.
- **Registry plus factory.** Robot, scene, task, policy, planner and backend registries map
  stable names to factories. `register_embodiment()` registers everything one robot owns from
  one `RobotDescriptor`. Builtins load lazily; a research module registers itself on import.
- **Composition root.** `physai.runtime.create_runtime()` (and `create_session()` for a
  manifest). `run_sim.py`, `eval_policy.py` and `collect_demos.py` use it; the other scripts
  hand-assemble an `EnvConfig` and stay safety-gated, since the gate lives in the adapter.
- **Safety gate.** `SafetyController` validates action mode, joint order, finite values,
  timestamps, joint limits and per-joint step limits immediately before a robot port receives
  a command. It lives inside the adapters (`DirectAdapter`, `MuJoCoROSBridge`, the Gymnasium
  adapter), so no workflow reaches the robot unchecked. A refused action raises
  `SafetyViolation` (a `ValueError`), which an evaluation records as an unsafe action.
  `RobotSpec.max_joint_delta` bounds a command against the *measured* position and must stay
  above `JointRateLimiter`'s command-to-command clamp, or normal servo lag trips the gate.

## Extension seams

Every seam is one new file plus one registration call.

| Seam | New file | Registration |
| --- | --- | --- |
| Robot | `robots/<name>/` package | One `RobotDescriptor` + `register_embodiment(name, descriptor)` in `robots/registry.py:_load_builtins()` |
| Scene | `sim/mujoco/scenes/<name>.py` | `register_scene(name, SceneDefinition(...))` in `sim/mujoco/scenes/registry.py:_load_builtins()` (declares supported robot kinds and tasks) |
| Task | `tasks/<name>.py` | `register_task(name, factory)` in `tasks/registry.py:_load_builtins()` |
| Policy (core baseline) | `policy/<name>.py` | `register_policy(name, factory)` in `policy/registry.py:_load_builtins()` |
| Policy (robot-owned or research) | `robots/<robot>/<name>.py` or `research/<topic>/<name>.py` | `register_robot_policy(robot, name, factory)` or `register_policy(name, factory)`; research calls it on import |
| Planner | `planner/base.py` (core) or `research/<topic>/<name>.py` | `register_planner(name, factory)` in `planner/registry.py:_load_builtins()`, or on import for research |
| Transport adapter | `robots/adapters.py` (a builder function) | `register_adapter(name, builder)` in `robots/adapters.py:_load_builtins()` |
| Simulator engine | `sim/<engine>/` package | Add the engine to a `RobotDescriptor.simulators` tuple; the robot's factory branches on `simulator=` |
| Client capability control | `web/static/js/controls.js` / `ui.js` | None: capabilities are declarative data (`RobotSpec.action_modes`/`capabilities`) |

A new robot also covers its capability contract and a generic simulation path in tests. A
frozen item (`contracts.py`, the ports in `robots/base.py`, `policy/base.py`, `tasks/base.py`,
`planner/base.py`, `SessionManifest`'s fields, the `Host` method surface, top-level package
boundaries) changes only with an entry in [DECISIONS.md](DECISIONS.md), a version marker on
the contract (for example `SessionManifest.schema_version`) and a deprecation window of one
minor version.

## Stable contracts

`Observation` carries joint state, named image frames, an optional end-effector pose, step
count and simulation time. `Action` carries either joint targets or a Cartesian/base `Twist`,
plus a normalized gripper command; the robot's capabilities decide which is valid, and an
action never carries both modes.

Camera resolution is chosen in one place: the manifest's `simulation.camera_resolution` or a
script's `--camera-res`, one of `physai.contracts.CAMERA_RESOLUTIONS` (320x240 default,
640x480, 1280x720). Scene configs, the Isaac env, the web host, the training contract and the
robot description read it, so results from different simulators see the same image.

`RobotSpec` describes an embodiment without exposing simulator or hardware API: joint names,
action modes, observation modalities, named capabilities, joint limits and optional per-joint
command-step limits. Workflow code calls `supports()`/`require()` instead of branching on a
robot name. Registration also declares an embodiment kind (`robot_kind()`) and robot-owned
`scene_defaults()`; scenes declare compatible robot kinds and task names, so an incompatible
combination fails during composition.

- `RobotPort` owns observation acquisition, action dispatch, lifecycle and the robot-specific
  mapping of joint names, units, gripper range, cameras and frames.
- `KinematicsPort` owns FK, IK, Jacobian and pinch-frame operations. Kinematics are
  embodiment-specific (the SO-101's is `robots/so101/kinematics.py`); tasks and policies
  request capabilities (`joint_position`, `arm_kinematics`, `base_velocity`, `odometry`),
  never robot names.

`Policy`, `PlanRunner` and task code depend on these ports, never on simulator types.
`SO101Env` is a robot-owned backend and creates no task; `TaskRuntime` wraps a robot port with
a registered task and owns reset, metrics, reward, success hold and termination.
`physai.policy.replay.VLAPolicy` (chunked-action shape) stays in core because the model-free
`ReplayPolicy` needs it; checkpoint-backed subclasses (`LeRobotPolicy`) are research.
`create_runtime` validates the task's required capabilities against the `RobotSpec`, resolves
the scene and wraps the robot with `TaskRuntime`; safety is the adapter's job, not its.

Registered compositions: `so101` + `single_cube_place` (or `sorting`) with the
scripted, visual-servo or Planner + `PlanRunner` policies, and `turtlebot4` with a constant
twist policy. TurtleBot4 proves the capability abstraction generalizes beyond an arm; it is
maintained, not a development focus ([ROADMAP.md](../ROADMAP.md)).

Task scenes build on `sim/workspace.py` (`WorkspaceConfig`); each scene describes its cubes
through `cubes()`, builds MuJoCo through `build_spec()`/`build_model()` and names its
arrangement through `layout_kind`, which the robot maps to a layout strategy
(`robots/so101/layout.py`). A new arrangement adds a layout, not an environment edit.

## Session manifest

One YAML file describes a run ([DECISIONS.md B](DECISIONS.md#b-the-manifest-is-the-only-run-description);
the field list and rules are in `physai/config/manifest.py`'s docstring). A single-robot run
is a manifest with one entry in `robots`; a `world` block (or several robots) makes one
shared MuJoCo world.

```yaml
schema_version: 1
simulation: {seed: 0}            # the one source of seed and randomization
scene:
  name: single_cube_place
  overrides: {target_radius: 0.04}
simulator: mujoco                # mujoco | isaac
backend: direct                  # direct | ros2_sim | ros2_real
task: single_cube_place
success_hold_steps: 10
robots:
  - id: arm_1
    robot: so101
    config: {max_steps: 400}     # robot env fields
    pose: {position: [0.0, 0.0, 0.0]}
    policy: scripted             # optional; "idle" if omitted
viewer: {mode: none}             # none | native | web | both
```

Validation resolves every `robot`, `task` and `policy` name through the registries and checks
scene, robot-kind and task compatibility without constructing a robot or model.
`create_session()` supports `backend: direct`. A non-MuJoCo engine rejects a `world` block,
several robots, `backend: ros2_sim` and any `viewer.mode` other than `none`;
`run_sim.py --sim {mujoco,isaac}` overrides `simulator` through
`config.compat.with_overrides`, which re-runs the validation. Worked examples are in
`configs/manifests/`.

### Scripts

`run_sim.py` (run and look: viewer, web host, video, recording), `eval_policy.py` (measure a
policy over N seeds, `--json`, difficulty flags) and `collect_demos.py` (scripted-expert
dataset) share the session and `physai.runtime.run_episode`, and differ only in what they
output. A script parses flags, builds a manifest with overrides, calls the rollout and prints
or saves the result; the output flags (`--video`, `--camera`, `--record`, `--out`, `--name`)
are defined once in `scripts/_common_args.py` and written by `scripts/_outputs.py`. Flag
names are one per concept and every script's `--help` follows `scripts/_cli.py`;
`tests/core/unit/test_cli_conventions.py` rejects retired names and flags without help.

## Host and client API

`physai.web.host.Host` is the one host class ([DECISIONS.md E](DECISIONS.md#e-one-host-a-frozen-desktop-viewer-record-and-replay-through-the-data-path)).
`Host.for_robot(...)` builds a direct-MuJoCo session (one `RobotPort`, optional policy) and
`Host.for_world(...)` a `SharedWorld` session (N namespaced instances, command and hold only).
Every method is instance-keyed with `instance_id` optional:

```text
Host
 ├── scene() / list_robots()               capability + geometry discovery
 ├── latest_state()                        dynamic transform snapshot
 ├── camera_jpeg(name, instance_id=None)   cached camera JPEG
 ├── submit(action, instance_id=None, ...) command + control lease
 ├── reset() / set_paused(paused)          world-atomic
 ├── start_recording() / stop_recording(success) / recording_status()
 ├── list_episodes() / load_episode(file) / seek() / set_playback() / exit_playback()
 ├── release_control(source)
 └── start() / stop()                      physics-thread lifecycle
```

One physics thread runs the control loop (30 Hz default); a separate camera worker renders
named cameras on its own cadence, so capture never pauses physics. `Host.physics_lock`
serializes MuJoCo access between them and the native `--viewer`. Manual control uses a short
per-instance lease: the first client to submit controls the instance until it disconnects or
the lease expires. `web/app.py` (FastAPI) is a thin client that never branches on host type
or imports `mujoco`; its routes (`/api/robots`, `/api/scene`, `/api/state`, `/api/episodes`,
`/api/mesh/{id}`, `/api/camera/{name}.jpg` and `/stream`, `WS /ws`) are listed in
`web/app.py`, and controls, recording and playback in the
[web viewer runbook](WEB_VIEWER_RUNBOOK.md). Shared-world instances are built through
`robots.registry.create_shared_instance()`; `SharedWorld` takes an optional `shared_attach`
hook so a robot can inject shared-world-only MJCF. Jogging is a capability a robot registers
(`RobotDescriptor.jog`, resolved by `create_jog_resolver()`), and the web client hides
controls a robot's capabilities do not support.

## Research boundary

Research plugs in through the existing contracts only (`Planner`, `Policy`, `Task`,
`Observation`/`Action`, `RobotTrainingContract`, `physai.data`, the Gymnasium adapter) and
registers itself on import ([DECISIONS.md A](DECISIONS.md#a-research-code-lives-outside-the-core-package),
[research/README.md](../research/README.md)). Core keeps minimal baselines (`ScriptedPlanner`,
`PlanRunner`, `ReplayPolicy`/`VLAPolicy`, `SortingTask`, `GymnasiumAdapter`, TurtleBot4
navigation); the scripted expert, visual servo, ACT/LeRobot pipeline and the sorting planner
are research. ACT is a low-level policy checkpoint format behind the policy boundary
(`LeRobotPolicy`), never a planner. `tests/core/` and `tests/research/<topic>/` run in
separate CI jobs so research dependencies (torch, LeRobot) stay isolated.

## Demonstration data

Demonstrations are LeRobot-shaped arrays; feature names, cameras, action layout and encoder
belong to the robot's `RobotTrainingContract` (`src/physai/robots/<robot>/contracts.py`). For
SO-101: `observation.images.front` and `.wrist` `(T, H, W, 3)` uint8, `observation.state`
`(T, 6)` float32 radians, `action` `(T, 6)` absolute joint targets, in the order
`shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`. `physai.data`
records and loads them without importing a concrete robot; the canonical schemas are
`so101_observation_spec()` and `so101_action_spec()`, and dataset metadata serializes them.
The recorder writes a compact `.npz` with LeRobot-shaped keys (a future `LeRobotDataset`
exporter must reuse the same metadata). A recording may add `observation.environment_state`
(`(T, nq)` float64, the full simulator `qpos`), declared in `meta.json`'s `features`; the web
recorder and `collect_demos.py` write it. Metadata also records the scene name and
configuration snapshot.

## ROS2 boundary

Direct mode runs in one process without ROS2. `physai/contracts.py` mirrors the ROS2 types
while staying transport-neutral, and `physai/bridge/ros2_contract.py` is the source of truth
for topics, message types, rates and frames. The synchronous bridge core and the `rclpy`
nodes target ROS2 Jazzy on Ubuntu 24.04. `ROS2SimAdapter` and `ROS2HardwareAdapter` subscribe
to the joint trajectory and gripper endpoints, decode them into the shared `Action`, expose
the latest complete command through a synchronous tick API and publish canonical state through
an injected `MessageCodec`; unit conversion, joint order, timestamps, frame names, command
freshness and rate are explicit and validated against `RobotSpec`. `rclpy` is imported lazily
inside each robot's `ros2_node.py`, which keeps `physai.sim` free of it.

Before hardware: (1) direct contract tests; (2) ROS2 simulator integration tests; (3)
hardware-driver tests with recorded or fake joint states; (4) a supervised smoke test with
command timeout, joint limits, emergency stop and stale-observation handling.

## Embodiment constraints and known gaps

The SO-101 has five arm degrees of freedom and no shoulder roll, so its planner uses
approach-constrained waypoints; table waypoints must stay near the tested surface and within
reach. The simulation uses the new-calibration SO-101 model; changing its files requires
rechecking pad geometry, joint limits, contact behavior and reachable area.

- `TwistToJointResolver` is built separately in `robots/so101/mujoco_env.py`,
  `robots/so101/shared.py` and `scripts/teleop_keyboard.py`, although the jog math is shared.
- `Host` branches on single-robot versus shared world in about a dozen places, because only a
  single robot runs a policy, records or plays back; shared-world sessions run no tasks or
  policies yet.
- Isaac is single-cube and observation-only ([DECISIONS.md D](DECISIONS.md#d-isaac-sim-is-an-optional-peer-engine));
  a manifest combining `simulator: isaac` with an unsupported feature is rejected at load.
  `collect_demos.py` takes no `--sim`.
- `SO101ROS2Node.handle_cartesian_target()` exists but is not bound to a ROS2 service.
