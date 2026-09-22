# Architecture

This is the internal design reference for the runtime. It defines the frozen
design shape, module ownership, dependency rules, stable contracts, the
session manifest, the host and client API, extension seams, the capability
model, and the research/core boundary. User setup and runnable commands live
in [README.md](../README.md). Agent workflow rules live in
[AGENTS.md](../AGENTS.md). Design decisions behind the frozen shape are
recorded as ADRs in [docs/adr/](adr/).

## Design shape

The stack separates embodiment, task, and decision-making so each can evolve
independently. A runtime composition selects one or more robots, one task
where applicable, and one planner or policy per session, then connects them
through message-shaped contracts.

```text
instruction + camera images
                    |
                    v
     Planner (scripted; model backends are research, not core)
                    |
                    v
     Plan: sub-goals + optional PoseStamped waypoints
                    |
                    v
     Policy or PlanRunner at the control rate
                    |
                    v
     Action selected by robot capabilities
                    |
                    v
     Robot environment and task evaluation
```

The current runnable baselines are the scripted planner/policy path with
`PlanRunner` and the SO-101 visual-servo policy; model-backed planners and VLA
policies plug into the same boundaries from `research/`.

## Execution modes and deployment parity

MuJoCo and ROS2 are different layers, not competing robot backends. MuJoCo
provides simulation and physics; ROS2 provides process and device transport.
The project supports three execution paths with different purposes:

```text
Fast development:
     Planner/Policy -> direct RobotPort -> MuJoCo

Integration validation:
     Planner/Policy node -> ROS2 topics -> ROS2MuJoCoAdapter -> MuJoCo

Real deployment:
     Planner/Policy node -> ROS2 topics -> ROS2HardwareAdapter -> hardware
```

The two ROS2 paths are the deployment-equivalent paths. They must expose the
same topic, message, unit, frame, joint-order, and rate contracts so that a
policy node can move from simulated hardware to real hardware without a policy
change. Direct MuJoCo is a fast path for unit tests, training, deterministic
regression tests, and physics debugging; it is not a substitute for ROS2
integration validation.

Implementation status: `DirectMuJoCoAdapter` and `ROS2MuJoCoAdapter` are
implemented. `ROS2HardwareAdapter` is a contract-only placeholder — simulation
only for now, per [ROADMAP.md](../ROADMAP.md); the adapter boundary below
exists so a real backend is additive when hardware work starts.

```text
RobotPort
     +-- DirectMuJoCoAdapter
     +-- ROS2MuJoCoAdapter
     +-- ROS2HardwareAdapter
```

The adapters are generic at the port level and receive an embodiment-specific
mapping and configuration. `DirectMuJoCoAdapter` and `ROS2MuJoCoAdapter` use
MuJoCo as the simulation engine, while `ROS2HardwareAdapter` connects the
selected embodiment to its real device driver. Backends are registry-driven
(`robots.registry.register_adapter()`/`create_adapter()`; `select_adapter()`
is the stable thin wrapper robot factories call) — adding a fourth backend is
additive, not a new branch inside an existing adapter.

`DirectMuJoCoAdapter` may retain a synchronous `reset`/`step` API for fast local
execution. The two ROS2 adapters may use asynchronous callbacks and queues
internally, but they must translate to the same `Observation` and `Action`
contracts at the application boundary.

### Mode trade-offs

| Mode | Primary value | Residual risk |
| --- | --- | --- |
| Direct MuJoCo | Fast, deterministic iteration and CI | Does not exercise ROS2 timing, QoS, TF, or serialization |
| MuJoCo through ROS2 | Tests the production message and process boundary before hardware | Adds setup, scheduling, latency, and transport failure modes |
| Hardware through ROS2 | Tests real sensors, actuators, calibration, and safety behavior | Slow, nondeterministic, hardware-dependent, and higher risk |

## Module ownership

| Module | Owns | Must not own |
| --- | --- | --- |
| `physai.contracts` | Shared `Observation`, `Action`, and ROS2-shaped value types | Robot-specific ordering or task rules |
| `physai.config` | Typed YAML configuration parsing (`legacy`: per-robot/per-world loaders; `manifest`: the session manifest), delegated to robot-owned config factories and the registries | Simulation behavior, robot construction, or task evaluation |
| `physai.robots` | Embodiment discovery, `RobotSpec`, robot ports, factories, environments, robot-specific adapters, backend registry (`adapters.py`), and shared-world instance/attach factories | Task reward, planner decisions, or model SDKs |
| `physai.tasks` | Task state, reset rules, reward, metrics, and termination | Robot internals or action generation |
| `physai.sim` | MuJoCo simulation core, generic scene primitives, task-specific scene builders, the shared multi-robot world, rendering, and simulation time | Robot-specific environment logic, ROS2 transport, QoS, callbacks, or any robot-name branch (`SharedWorld` takes an optional `shared_attach` hook instead) |
| `physai.planner` | Instruction and image grounding, `Plan`, `SubGoal` production, and the planner registry | Control-rate motor commands; research planners |
| `physai.policy` | Control-rate `Action` production, core baseline policies (`constant`, `constant_twist`, `replay`), and the policy registry | Task scoring, robot discovery, or checkpoint-backed inference |
| `physai.control` | Action resolution, capability checks, and rate limiting | High-level planning or task semantics |
| `physai.data` | Episode recording, dataset/checkpoint metadata, evaluation, and the Gymnasium adapter | Simulation decisions or model inference |
| `physai.bridge` | ROS2 transport, topic/message mapping, timing, and ROS2-backed adapters | Physics implementation, task semantics, or model inference |
| `physai.runtime` | Runtime composition, compatibility checks, and safety orchestration | Robot-specific physics, task reward, or model inference |
| `physai.web` | The one `Host` class, the FastAPI client surface, telemetry payloads, and the static browser client | Simulator selection or robot construction (the composition root builds what a `Host` wraps) |
| `research/<topic>/` | Approach-specific implementations (scripted experts, classical control, imitation learning, reinforcement learning, VLA, VLM planners), each self-registering with a core registry on import | Anything a core module imports — see [Research boundary](#research-boundary) |
| `scripts/` | Generic CLI argument parsing and runtime composition | IK, reward calculation, or SDK-specific implementation |
| `launch/` | Generic ROS2 launch composition and launch arguments | Robot physics, transport callbacks, or task rules |
| `tests/` | Executable behavior and contract coverage | New runtime ownership |

The source tree follows this ownership map:

```text
src/physai/
├── contracts.py       shared message-shaped values
├── config/            typed YAML runtime configuration (legacy.py, manifest.py)
├── robots/            embodiment ports, adapters, registries, and factories
│   ├── so101/         SO-101 environment, kinematics, and shared-world adapter
│   └── turtlebot/     TurtleBot4 environment and shared-world adapter
├── tasks/             task rules and registry
├── sim/               MuJoCo simulation core, shared world, and scene orchestration
│   └── scenes/        shared world builder and task-specific scene variants
├── planner/           Planner contract, ScriptedPlanner baseline, and registry
├── policy/            Policy contract, core baselines, and registry
├── control/           action resolution and rate limiting
├── data/              episode recording, metadata, evaluation, and Gymnasium adapter
├── bridge/            ROS2 transport, message mapping, adapters, and tick loop
├── runtime/           robot-task-policy composition and safety orchestration
└── web/               the one Host class, FastAPI app, telemetry, static client

research/
├── scripted_experts/      privileged-ground-truth demo-collection policies
├── classical_control/     visual servo and other model-free baselines
├── imitation_learning/    ACT/LeRobot dataset tooling, training, checkpoints
├── reinforcement_learning/  deep RL on the Gymnasium adapter (placeholder)
├── vla/                   VLA research beyond the checkpoint adapter (placeholder)
└── vlm_planners/          model- or heuristic-grounded Planner implementations
```

## Dependency direction

These four rules are enforced by `uv run lint-imports`
(`pyproject.toml`'s `[tool.importlinter]`) and run inside the normal test
suite via `tests/boundaries/test_import_boundaries.py`, so a violation fails
`pytest tests/ -q` the same way a failing unit test would.

| Rule | Rationale |
| --- | --- |
| `physai.sim` must not import `physai.bridge` or `rclpy` | Keeps ROS2 optional for fast, deterministic MuJoCo-only workflows |
| `physai.policy`, `physai.tasks`, `physai.planner` must not import `mujoco` | Policy/task/planner code depends on ports (`RobotPort`, `RobotSpec`), never on MuJoCo types. One accepted, documented indirect path exists: `policy.registry` calls `robots.registry.create_robot_policy()` to build a robot-owned policy without importing the robot itself; `robots.registry`'s own lazily loaded robot implementations are what actually touch MuJoCo, and policy never inspects the returned object's type. |
| `physai` (core) must not import `research` | Research plugs in by registering itself into a core registry on import; core never reaches back into it. See [Research boundary](#research-boundary). |
| A robot-name branch in generic code is a bug | `SharedWorld`, `web.host.Host`, and the web static client must resolve behavior through `RobotSpec` capabilities or a registry lookup, never `if robot_name == "..."`. Not import-linter-checked (it is a code-shape rule, not an import-graph rule); enforced by review. |

## Design patterns in use

- **Ports and Adapters:** `Observation`, `Action`, `RobotPort`, `Planner`,
  `Policy`, and `Task` are ports. Direct MuJoCo, ROS2 transport, model SDKs,
  and concrete robot implementations are adapters around those ports.
- **Strategy:** planners, policies, and tasks are interchangeable strategies.
  The caller depends on the abstract contract and selects the behavior at
  runtime.
- **Registry plus Factory:** robot, scene, task, policy, planner, and backend
  registries map stable names to factories. `robots.registry.register_embodiment()`
  registers everything one robot owns (factory, scene defaults, env config,
  ROS2 node, navigation, shared-world attach/instance) in one call, given one
  `RobotDescriptor`. Lazy builtin loading keeps optional dependencies out of
  model-free workflows and lets extensions register without editing a caller;
  a research module registers itself on import instead of a core registry
  importing it.
- **Composition Root:** `physai.runtime.create_runtime()` is the intended
  composition root; CLI modules under `scripts/` are meant to call it and
  parse arguments only. `run_sim.py` calls it for the manifest-free
  single-robot path's robot construction but still assembles the task/policy
  and (for `--world`) the shared world by hand — see
  [Known remaining gaps](#known-remaining-gaps).
- **Safety Gate:** `SafetyController` validates action mode, joint order,
  finite values, timestamps, joint limits, and configured per-joint step
  limits immediately before a robot port receives a command. The gate lives
  inside the adapters (`DirectMuJoCoAdapter`, `MuJoCoROSBridge`, the
  Gymnasium adapter), so every execution path shares it and no workflow can
  reach the robot unchecked. `RobotSpec.max_joint_delta` bounds command
  against *measured* position and must stay above `JointRateLimiter`'s
  command-to-command clamp, or normal servo tracking lag trips the gate.

These patterns are intentionally lightweight. A new abstraction is warranted
only when it removes coupling at a boundary or makes a component replaceable;
inheritance should not be added solely to make a class hierarchy.

## Extension seams

Every seam is one new file plus one registration call, except where noted.

| Seam | New file | Registration |
| --- | --- | --- |
| Robot | `robots/<name>/` package | One `RobotDescriptor` + `register_embodiment(name, descriptor)` in `robots/registry.py:_load_builtins()` |
| Scene | `sim/scenes/<name>.py` | `register_scene(name, SceneDefinition(...))` in `sim/scenes/registry.py:_load_builtins()` |
| Task | `tasks/<name>.py` | `register_task(name, factory)` in `tasks/registry.py:_load_builtins()` |
| Policy (core baseline) | `policy/<name>.py` | `register_policy(name, factory)` in `policy/registry.py:_load_builtins()` |
| Policy (robot-owned) | `robots/<robot>/<name>.py` or `research/<topic>/<name>.py` | `register_robot_policy(robot_name, policy_name, factory)`, called by the module itself if it lives in `research/` |
| Policy (research, global) | `research/<topic>/<name>.py` | `register_policy(name, factory)`, called by the module itself on import |
| Planner (core baseline) | `planner/base.py` | `register_planner(name, factory)` in `planner/registry.py:_load_builtins()` |
| Planner (research) | `research/<topic>/<name>.py` | `register_planner(name, factory)`, called by the module itself on import |
| Backend | `robots/adapters.py` (a new builder function) | `register_adapter(name, builder)` in `robots/adapters.py:_load_builtins()` |
| Client capability control | `web/static/js/controls.js` / `ui.js` | None — capabilities are declarative data (`RobotSpec.action_modes`/`capabilities`), rendered conditionally; this is a UI branch on data, not a registry |

Robot registration touches one function in one file (`_load_builtins()`),
even though a robot with every optional factory (env config, ROS2 node,
navigation, shared-world attach/instance) supplies up to six fields on one
`RobotDescriptor` — still one call, not one call per factory kind.

## Frozen vs. free to change

| Layer | Frozen (needs an ADR + version bump) | Free to change |
| --- | --- | --- |
| Contracts | `contracts.py` (`Observation`, `Action`, `*Spec`, `Header`, `JointState`, `Pose`, `Twist`, ...) | Internal helpers not part of the public dataclass shape |
| Ports | `robots/base.py` (`RobotPort`, `RobotSpec`, `RobotTrainingContract`, `KinematicsPort`); `policy/base.py` (`Policy`); `tasks/base.py` (`Task`); `planner/base.py` (`Planner`, `Plan`/`SubGoal`) | Concrete implementations of each ABC |
| Manifest schema | `SessionManifest`'s field names/types/validation rules (see [Session manifest](#session-manifest)) once a session depends on them | Which YAML files exist under `configs/` |
| Host API | The method surface in [Host + client API](#host--client-api) | `Host`'s internal threading model, camera cadence, lease timeout |
| Folder tree | Top-level package boundaries (`robots/`, `policy/`, `tasks/`, `planner/`, `sim/`, `web/`, `runtime/`, `research/`) and the rule that core never imports `research/` | Internal file layout inside one robot's package or one research topic |
| Registries | The register/create pattern itself | Which specific factories are registered, and in what order |

Evolution rule: a frozen item changes only via (1) an ADR under `docs/adr/`,
(2) a version marker on the affected contract (e.g. `SessionManifest.schema_version`),
and (3) a deprecation window — old names/routes keep working with a warning
for at least one minor version before removal.

## Stable contracts

`Observation` contains joint state, named image frames, optional end-effector
pose, step count, and simulation time. `Action` can carry either joint targets
or a Cartesian/base `Twist`, plus a normalized gripper command. The selected
robot capability decides which representation is valid; an action must not
silently carry both modes.

`RobotSpec` describes an embodiment without exposing simulator or hardware API:
its joint names, action modes, observation modalities, named capabilities,
joint limits, and optional per-joint command-step limits.
Workflow code should call `supports()` or `require()` rather than branch on a
robot name.

Robot registration also declares an embodiment kind. The registry exposes
`robot_kind()` and robot-owned `scene_defaults()` so runtime composition can
select and validate scenes before constructing a concrete robot. Scene
definitions declare compatible robot kinds and task names; incompatible
combinations fail during composition rather than inside a policy or scene
builder.

The robot boundary consists of two explicit ports around the existing
contracts:

- `RobotPort` owns observation acquisition, action dispatch, lifecycle, and
  the robot-specific mapping of joint names, units, gripper range, cameras,
  and frames.
- `KinematicsPort` owns FK, IK, Jacobian, and pinch-frame operations. A
  MuJoCo implementation may use `MjModel`/`MjData`; a hardware implementation
  must consume measured joint state and a calibrated kinematics model instead.

`Policy`, `PlanRunner`, and task code should depend on these ports rather than
on MuJoCo types. `SO101Env` is a robot-owned backend that provides observation,
action, lifecycle, and embodiment state; it does not create or evaluate a
task. `TaskRuntime` composes a registered task around a robot port and owns
task reset, metrics, reward, success hold, and termination for synchronous
direct-MuJoCo workflows.

`Planner` maps an instruction and observation to `Plan`. A `Plan` contains
language-grounded `SubGoal` values and optional `PoseStamped` waypoints.
`Policy` maps an observation and optional goal to one `Action` per control
tick. `Task` owns evaluation, reward, and termination around the backend
state.

`physai.policy.replay.VLAPolicy` is the chunked-action `Policy` shape
(observation packing, action-chunk buffering, unit decoding) shared by any
policy that consumes a chunk of actions at a time; it stays in core because
`ReplayPolicy` — a model-free integration test for the whole seam — needs it,
and core must not import research. Checkpoint-backed subclasses
(`LeRobotPolicy`) are research content: see
[Research boundary](#research-boundary).

`physai.runtime.create_runtime` composes a runtime in one call: it validates
the task's declared capabilities against the selected `RobotSpec`, resolves the
scene, wraps the robot port with `TaskRuntime` when a task is selected, and
creates an optional registered policy. Safety is not its job — the adapter
gates every action regardless of how the runtime was assembled.

Scene configs are selected through `physai.sim.scenes.create_scene` and the
scene registry. Each registered scene declares supported robot kinds and task
names. `create_runtime(scene_name=...)` resolves that config before building
the robot and rejects task-scene mismatches during composition. A scene's
embodiment defaults come from the robot's `scene_defaults()` and are passed in
explicitly; the scene layer never looks up a robot by name.

### Task-specific scenes

Scene geometry is split by task while robot and workspace components are shared:

```text
sim/scenes/common.py
     +-- WorldSceneConfig: generic model, workspace, and camera settings
     +-- ManipulationSceneConfig: configurable end-effector and pad attachments
     +-- shared manipulation-world builder

sim/scenes/pick_place_minimal.py
     +-- one cube and one target layout

sim/scenes/sorting_minimal.py
     +-- colored cube layout and sorting positions
```

Each scene builder owns model geometry and initial object layout. Robot model
paths and end-effector anchors are configuration, not hardcoded task
ownership. Each scene config builds itself through `build_spec()`/
`build_model()`, so selecting a scene is selecting a class rather than
decoding an object-count flag.

## Session manifest

One YAML file binds robot instances, scene, task, policy, backend, and viewer
options for a run. A single-robot run is a manifest with one entry in
`robots`. Loaded and validated by `physai.config.manifest.load_manifest()`
(full field-by-field schema and validation rules are documented in that
module's docstring — this section is the summary):

```yaml
schema_version: 1
scene:
  name: pick_place_minimal
backend: direct                  # direct | ros2_sim | ros2_real
robots:
  - id: arm_1
    robot: so101
    pose:
      position: [0.0, 0.0, 0.0]
      quaternion: [1.0, 0.0, 0.0, 0.0]
    task: pick_place              # optional per-instance override
    policy: scripted               # optional; "idle" if omitted
task: pick_place                  # optional session-wide default
policy: idle
viewer:
  mode: none                      # none | tk | web | both
```

Validation resolves every `robot`/`task`/`policy` name through the existing
registries and checks scene/robot-kind/task compatibility via
`SceneDefinition.supports()` and `robot_kind()` — without constructing any
robot or MuJoCo model. `backend: ros2_real` is accepted by the schema for
forward compatibility but raises a clear "not yet implemented" error.

`configs/manifests/example_single_so101.yaml` and
`example_heterogeneous.yaml` are worked examples.
`physai.config.legacy` (`load_sim_config`/`load_task_config`/
`load_world_config`, backing `configs/sim_config.yaml`,
`configs/tasks/<robot>/*.yaml`, and `configs/worlds/*.yaml`) is unaffected
and stays for CLI back-compat — see
[Known remaining gaps](#known-remaining-gaps) for what still uses it instead
of the manifest.

## Host + client API

`physai.web.host.Host` is the one host class. Build one with `Host.for_robot(...)`
(direct MuJoCo, one `RobotPort`, optional policy) or `Host.for_world(...)`
(a `SharedWorld` with N namespaced instances, command/hold only, no policy) —
a single-robot session is simply a `Host` with one instance. Every method is
instance-keyed with `instance_id` optional, defaulting to the sole instance:

```text
Host
 ├── scene() / list_robots()               capability + geometry discovery
 ├── latest_state()                        dynamic transform snapshot
 ├── camera_jpeg(name, instance_id=None)   cached camera JPEG
 ├── submit(action, instance_id=None, ...) command + control lease
 ├── reset() / set_paused(paused)          world-atomic, affects every client
 ├── release_control(source)               drop every instance a source owns
 └── start() / stop()                      physics-thread lifecycle
```

One physics thread runs the control loop (30 Hz by default); a separate
camera worker thread renders named cameras at a slower, decoupled cadence
(0.2 s) so camera capture never pauses the physics loop. `Host.physics_lock`
serializes MuJoCo access between the two threads and any renderer client
(e.g. the Tk viewer). HTTP camera requests only read the latest cached JPEG;
they never step or render from the request handler itself.

Reset and pause are world-atomic and affect every connected client. Manual
control uses a short per-instance lease: the first client to submit a command
becomes that instance's controller; another client's command is rejected
while the lease is alive. Releasing the connection or letting the lease
expire returns that instance to its hold action or configured policy.
`release_control(source)` only drops the instances that `source` actually
owns, so one client's disconnect never cancels another instance's rightful
owner mid-lease.

`web/app.py` (FastAPI) is a thin client of `Host` — it never branches on host
type, and it never imports `mujoco` or a robot module. The REST/WebSocket
surface is stable across single- and multi-robot sessions:

| Route | Behavior |
| --- | --- |
| `GET /` | Three.js browser console (static file) |
| `GET /api/robots` | Every instance's id, kind, action modes, capabilities, and cameras |
| `GET /api/scene` | Static geometry manifest for the whole session |
| `GET /api/state` | Latest transform snapshot for the whole session |
| `GET /api/mesh/{id}` | Compiled mesh binary payload |
| `GET /api/camera/{name}.jpg` | Cached camera JPEG (single snapshot), optionally `?robot=<instance_id>` |
| `GET /api/camera/{name}/stream` | MJPEG `multipart/x-mixed-replace` stream; 404 on an unknown camera |
| `WS /ws` | Accepts `select_robot`, `command`, `reset`, `pause`, `release_control`; streams state + errors |

Both clients depend only on this API plus `RobotSpec` capabilities, never on
MuJoCo or robot internals:

- **`--viewer` (Tk):** frozen scope — simulation render, live camera panels,
  basic status. It never grows features (see
  [ADR 4](adr/0004-tk-viewer-frozen.md)). Its camera-panel list is driven by
  `RobotSpec.camera_frames`, not by introspecting the MuJoCo model; the free
  3D scene view still uses `mujoco.Renderer` directly against `Host.model`,
  which is inherent to rendering the physics scene itself.
- **`--serve` (FastAPI + Three.js):** the sophisticated client. All new UI
  features (jog controls, instance selection, telemetry, overlays) go here.
  The browser talks only to the REST/WebSocket surface above; see
  [docs/WEB_VIEWER_RUNBOOK.md](WEB_VIEWER_RUNBOOK.md) for keyboard mapping,
  cloud-workspace setup, and troubleshooting.

Shared-world instances (`SharedWorldInstance`-shaped adapters returned by
`robots.registry.create_shared_instance()`) are built entirely through the
robot registry — `web/host.py` and `sim/world.py` never branch on a robot
name; `sim.world.SharedWorld` takes an optional `shared_attach` callable
(wired to `robots.shared_attach`) so a robot can inject shared-world-only
MJCF (e.g. SO-101's extra cameras) without `sim` knowing any robot's name.

## Capabilities and kinematics

Kinematics are embodiment-specific. The SO-101 implementation in
`src/physai/robots/so101/kinematics.py` owns FK, IK, Jacobian, and gripper
conversion for that arm; it is not a generic kinematics service.

```text
capability contract
                    |
                    +-- SO101Kinematics       arm FK, IK, Jacobian, gripper
                    +-- TurtleBotMotion       base velocity, odometry
                    +-- MobileManipulatorIK   base motion, arm IK, gripper
```

A mobile base may expose `base_velocity` and `odometry`; an arm may expose
`joint_position` and `arm_kinematics`; a mobile manipulator may expose both.
Tasks and policies request capabilities, not robot names.

Jogging is meant to be a capability a robot owns and registers (a resolver
factory alongside its other `RobotDescriptor` fields), with the host exposing
one generic jog command and UIs rendering controls from `action_modes`/
`capabilities`. The UI side of this is already true (the web client hides
controls a robot's capabilities don't support). The resolver-ownership side
is not finished — see [Known remaining gaps](#known-remaining-gaps).

## Research boundary

Research plugs in through the existing contracts only (`Planner`, `Policy`,
`Task`, `Observation`/`Action`, `RobotTrainingContract`, `physai.data`, the
Gymnasium adapter) and registers itself into a core registry on import. Core
ships the contracts plus minimal baselines only; per-module placement:

| Module | Home | Why |
| --- | --- | --- |
| `ScriptedPlanner`, `PlanRunner`, `ReplayPolicy`/`VLAPolicy` base, `SortingTask`, `GymnasiumAdapter`, TurtleBot4 `navigation.py` | Core | Minimal, technique-agnostic baselines the brief treats as core's job to ship |
| `research/scripted_experts/so101_pick_place_expert.py` | Research | Privileged ground-truth expert (reads `mujoco` object pose directly) |
| `research/classical_control/so101_visual_servo.py` | Research | One calibrated-camera visual-servo baseline, not infrastructure |
| `research/imitation_learning/{act_dataset,vla_adapter,train_act}.py` | Research | ACT/LeRobot training pipeline and the checkpoint-backed `LeRobotPolicy` |
| `research/vlm_planners/sorting_planner.py` | Research | Task-coupled (reads privileged `env.cube_positions`), not a generic baseline |
| `research/reinforcement_learning/`, `research/vla/` | Research (placeholders) | No concrete module yet; topics reserved per [ROADMAP.md](../ROADMAP.md) |

See [ADR 3](adr/0003-research-outside-core.md) for the decision and
`research/README.md` for the one rule research code follows.

## Runtime compositions

### Implemented compositions

```text
so101 + pick_place + scripted, visual-servo, or Planner + PlanRunner + MuJoCo
turtlebot4 + generic smoke test + constant twist policy + MuJoCo
```

`so101` and `turtlebot4` are examples of registered embodiments, not names
that belong in the generic adapter or policy contracts. `pick_place` is the
registered manipulation task and requires arm and gripper capabilities, so the
policy, demo, and planner workflows are currently SO-101-specific. TurtleBot4
has a native MuJoCo model, differential-drive controls, wheel state, base pose,
direct RPP navigation, and a ROS2/Nav2 obstacle acceptance path. It proves the
capability abstraction generalizes beyond one arm and is explicitly not a
development focus (see [ADR 5](adr/0005-turtlebot4-kept-as-second-embodiment.md)).

The current pick-and-place and sorting implementations are intentionally
minimal baselines for smoke tests and early experiments. They live in
`physai.tasks.pick_place_minimal` and `physai.tasks.sorting_minimal`, where
`SortingTask` extends `PickPlaceTask` with the target-color state; the
registry keys `pick_place` and `sorting` remain stable so configuration does
not encode an implementation filename.

## Model roles

No model-backed planner is built in core. `physai.planner` defines the
`Planner` contract and a scripted backend; `research/vlm_planners/` is where
a VLM-grounded backend implementing the same contract belongs.

ACT is the low-level policy checkpoint format behind the policy boundary in
`research/imitation_learning/vla_adapter.py` (`LeRobotPolicy`). A policy
checkpoint predicts control-rate actions; it must not be described as a
planner or be coupled directly to a robot environment.

Checkpoints are written by `research/imitation_learning/train_act.py` into
the ignored local `outputs/` directory and loaded through an explicit path.

## Demonstration data

Robot demonstrations use LeRobot-shaped arrays. The feature names, camera
streams, action layout, and encoder belong to the selected robot's
`RobotTrainingContract` in `src/physai/robots/<robot>/contracts.py`:

```text
observation.images.front  (T, H, W, 3) uint8
observation.images.wrist  (T, H, W, 3) uint8
observation.state         (T, 6) float32, radians
action                    (T, 6) float32, absolute joint targets
```

The six values are ordered as:
`shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`.
Dataset recording and loading belong to `physai.data`; task semantics do not.
The recorder consumes the robot contract without importing a concrete robot.

The canonical SO-101 schemas are `so101_observation_spec()` and
`so101_action_spec()` in `physai.robots.so101.contracts`; TurtleBot4 provides
the corresponding twist and wheel-state schemas in
`physai.robots.turtlebot.contracts`. Dataset metadata serializes these robot
contracts rather than defining a second joint or camera layout. The current
recorder writes a compact internal `.npz` format with LeRobot-shaped feature
keys; a future standard `LeRobotDataset` exporter must consume the same
metadata and must not introduce a parallel action contract.

Dataset and checkpoint metadata also record the selected scene name and scene
configuration snapshot. This makes training and evaluation reproducible when
the scene registry grows, while keeping scene construction owned by
`physai.sim.scenes`.

## ROS2 boundary

Direct MuJoCo can run in one process without ROS2. The shared values in
`src/physai/contracts.py` intentionally mirror the ROS2 types while remaining
transport-neutral; topic mapping is documented in
`src/physai/bridge/ros2_contract.py`. The synchronous bridge core and real
`rclpy` nodes are available for ROS2 Jazzy on Ubuntu 24.04.

The ROS2 boundary has two interchangeable adapter roles:

- `ROS2MuJoCoAdapter` translates the ROS2 topics into MuJoCo control and
  publishes simulated joint state and available camera frames.
- `ROS2HardwareAdapter` translates the same ROS2 topics into the selected
  embodiment's motor and camera interfaces and publishes measured state.

Both adapters have embodiment-specific implementations. The SO-101 is one
such implementation, not part of the generic adapter contract. Each adapter
subscribes to the joint trajectory and gripper command endpoints, decodes
those messages into the shared `Action` contract, and exposes the latest
complete command through its synchronous tick API. It publishes canonical
joint states and camera frames through an injected `MessageCodec`; the
default `ContractMessageCodec` is used by transport-neutral tests, while
`ROS2MessageCodec` can construct real ROS2 message instances without adding
`rclpy` as a core dependency.

Both adapters must make unit conversion, joint ordering, timestamps, frame
names, command freshness, and command rate explicit. Joint order and value
shape are validated at decode time against `RobotSpec`. The ROS2 contract file
is the source of truth for external interfaces; it is not itself an adapter.
`rclpy` is imported lazily inside each robot's `ros2_node.py`, only when its
node actually runs, which is how `physai.sim` stays free of it (enforced by
the import-boundary contract above) even though `robots.registry` eagerly
imports `ros2_node.py` modules that reference it.

### Required validation gates

Before hardware deployment, the following checks should pass:

1. Direct MuJoCo contract tests for `Observation`, `Action`, capabilities,
   joint ordering, limits, and gripper normalization.
2. ROS2 MuJoCo integration tests using the same topics and message types as
   the hardware driver.
3. Hardware-driver tests with recorded or fake joint states and camera frames.
4. A supervised hardware smoke test with command timeout, joint limits,
   emergency stop, and stale-observation handling.

## Embodiment constraints

The SO-101 model has five arm degrees of freedom and no shoulder roll, so its
planner uses approach-constrained waypoints instead of full six-degree-of-
freedom poses. Table waypoints must remain near the tested surface and within
the reachable workspace.

The simulation uses the new-calibration SO-101 model. Changes to robot model
files require rechecking pad geometry, joint limits, contact behavior, and
reachable area.

## Known remaining gaps

Tracked here rather than silently left implicit, since this document is
meant to be the frozen reference:

- **Jog resolver ownership.** `TwistToJointResolver` construction is
  duplicated across `web/host.py`'s single- and shared-world branches and
  `scripts/teleop_keyboard.py`, instead of each robot registering its own jog
  resolver factory on `RobotDescriptor`. The client-facing capability model
  (UIs render controls from `RobotSpec`) is already correct; only the
  resolver-construction side needs the extra registry field.
- **`scripts/` composition-root adoption.** Most scripts other than
  `run_sim.py`'s robot construction still hand-assemble `EnvConfig`/env
  objects instead of calling `physai.runtime.create_runtime()` — see
  [docs/MIGRATION.md](MIGRATION.md) for the per-script list. They are
  safety-gated either way (the gate lives in the adapter, not the
  composition path).
- **Session manifest adoption.** `physai.config.manifest.load_manifest()` is
  implemented and tested but not yet wired into `run_sim.py` or
  `create_runtime()`; `--config`/`--world` and their legacy loaders are still
  the only manifest-shaped configuration a script actually consumes.
- **Registry granularity.** Adding a robot is "one `RobotDescriptor` + one
  `register_embodiment()` call," not literally one line — a robot supplying
  every optional factory sets up to six fields on that one descriptor.
