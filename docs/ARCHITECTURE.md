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

MuJoCo simulates; ROS2 transports. They are different layers, not competing
backends, and three paths use them:

```text
Fast development:        Planner/Policy -> direct RobotPort -> MuJoCo
Integration validation:  Planner/Policy node -> ROS2 topics -> ROS2MuJoCoAdapter -> MuJoCo
Real deployment:         Planner/Policy node -> ROS2 topics -> ROS2HardwareAdapter -> hardware
```

The two ROS2 paths must expose the same topic, message, unit, frame,
joint-order, and rate contracts so a policy node moves from simulation to
hardware unchanged. Direct MuJoCo is the fast path for unit tests, training,
and regression; it does not replace ROS2 integration validation.

`RobotPort` has three adapters: `DirectMuJoCoAdapter` and `ROS2MuJoCoAdapter`
are implemented, and `ROS2HardwareAdapter` is a contract-only placeholder
(simulation only for now, per [ROADMAP.md](../ROADMAP.md)). Adapters are
generic at the port level and receive an embodiment-specific mapping.
Backends are registry-driven (`robots.registry.register_adapter()`/
`create_adapter()`; `select_adapter()` is the thin wrapper robot factories
call), so a new backend is additive. The direct adapter keeps a synchronous
`reset`/`step`; the ROS2 adapters may use callbacks and queues internally but
translate to the same `Observation` and `Action` at the boundary.

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
│   ├── so101/         SO-101 environment, object layouts, kinematics, and shared-world adapter
│   └── turtlebot/     TurtleBot4 environment and shared-world adapter
├── tasks/             task rules and registry
├── sim/               MuJoCo simulation core, shared world, and scene orchestration
│   └── scenes/        shared world builder and task-specific scene variants
├── planner/           Planner contract, ScriptedPlanner baseline, and registry
├── policy/            Policy contract, core baselines, and registry
├── control/           action resolution and rate limiting
├── data/              episode recording, metadata, evaluation, and Gymnasium adapter
├── bridge/            ROS2 transport, message mapping, adapters, and tick loop
├── runtime/           robot-task-policy composition and manifest sessions
└── web/               the one Host class (plus its control lease and camera worker), FastAPI app, telemetry, static client

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
  `Policy`, and `Task` are ports; direct MuJoCo, ROS2 transport, model SDKs,
  and concrete robots are adapters around them.
- **Strategy:** planners, policies, and tasks are interchangeable behind
  their abstract contracts.
- **Registry plus Factory:** robot, scene, task, policy, planner, and backend
  registries map stable names to factories.
  `robots.registry.register_embodiment()` registers everything one robot owns
  from one `RobotDescriptor`. Lazy builtin loading keeps optional
  dependencies out of model-free workflows; a research module registers
  itself on import instead of a core registry importing it.
- **Composition Root:** `physai.runtime.create_runtime()` is the intended
  composition root (`physai.runtime.create_session()` builds a whole manifest
  on top of it), and CLIs under `scripts/` should only parse arguments and call
  it. `run_sim.py` does; other scripts do not yet, see
  [Known remaining gaps](#known-remaining-gaps).
- **Safety Gate:** `SafetyController` validates action mode, joint order,
  finite values, timestamps, joint limits, and per-joint step limits
  immediately before a robot port receives a command. It lives inside the
  adapters (`DirectMuJoCoAdapter`, `MuJoCoROSBridge`, the Gymnasium adapter),
  so no workflow reaches the robot unchecked. `RobotSpec.max_joint_delta`
  bounds a command against the *measured* position and must stay above
  `JointRateLimiter`'s command-to-command clamp, or normal servo lag trips the
  gate.

These patterns are deliberately lightweight: add an abstraction only when it
removes coupling at a boundary or makes a component replaceable.

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

`Observation` carries joint state, named image frames, an optional
end-effector pose, step count, and simulation time. `Action` carries either
joint targets or a Cartesian/base `Twist`, plus a normalized gripper command;
the robot's capabilities decide which is valid, and an action never carries
both modes.

`RobotSpec` describes an embodiment without exposing simulator or hardware
API: joint names, action modes, observation modalities, named capabilities,
joint limits, and optional per-joint command-step limits. Workflow code calls
`supports()`/`require()` instead of branching on a robot name. Registration
also declares an embodiment kind (`robot_kind()`) and robot-owned
`scene_defaults()`; scene definitions declare compatible robot kinds and task
names, so an incompatible combination fails during composition rather than
inside a policy or scene builder.

Two ports surround these contracts:

- `RobotPort` owns observation acquisition, action dispatch, lifecycle, and
  the robot-specific mapping of joint names, units, gripper range, cameras,
  and frames.
- `KinematicsPort` owns FK, IK, Jacobian, and pinch-frame operations. A
  MuJoCo implementation may use `MjModel`/`MjData`; a hardware implementation
  must use measured joint state and a calibrated model.

`Policy`, `PlanRunner`, and task code depend on these ports, never on MuJoCo
types. `SO101Env` is a robot-owned backend (observation, action, lifecycle,
embodiment state) and creates no task. `TaskRuntime` wraps a robot port with a
registered task and owns reset, metrics, reward, success hold, and
termination for direct-MuJoCo workflows. `Planner` maps an instruction and
observation to a `Plan` of `SubGoal`s and optional `PoseStamped` waypoints;
`Policy` maps an observation and optional goal to one `Action` per control
tick.

`physai.policy.replay.VLAPolicy` is the chunked-action `Policy` shape
(observation packing, chunk buffering, unit decoding). It stays in core
because the model-free `ReplayPolicy` needs it; checkpoint-backed subclasses
(`LeRobotPolicy`) are research.

`physai.runtime.create_runtime` validates the task's required capabilities
against the `RobotSpec`, resolves the scene, wraps the robot with
`TaskRuntime` when a task is selected, and creates an optional registered
policy. Safety is not its job: the adapter gates every action.
`create_runtime(scene_name=...)` rejects task-scene mismatches during
composition, and a scene's embodiment defaults come from the robot's
`scene_defaults()`, so the scene layer never looks up a robot by name.

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

Each scene builder owns model geometry, and names its object arrangement
through a class-level `layout_kind`; the robot maps that name to a layout
strategy (`robots/so101/layout.py`) that places the objects each episode and
reads them back, so a scene with a new arrangement adds a layout rather than
editing the environment. Robot model paths, end-effector anchors, and the
gripper-specific pad and wrist-camera fit are configuration the robot supplies
through `scene_defaults()`, not generic defaults. Each scene config builds
itself through `build_spec()`/`build_model()`, so selecting a scene is
selecting a class rather than decoding an object-count flag.

## Session manifest

One YAML file describes a run: robot instances, scene, task, policy, backend,
simulation settings, and viewer options. A single-robot run is a manifest with
one entry in `robots`; a `world` block (or several robots) makes it one shared
MuJoCo world. `physai.config.manifest.load_manifest()` validates it (the field
list and validation rules are in that module's docstring) and
`physai.runtime.create_session()` builds it:

```yaml
schema_version: 1
simulation: {seed: 0}            # the one source of seed and randomization
scene:
  name: pick_place_minimal
  overrides: {camera_width: 640}  # scene fields, including robot_xml
backend: direct                  # direct | ros2_sim | ros2_real
task: pick_place                 # optional session-wide default
success_hold_steps: 10
robots:
  - id: arm_1
    robot: so101
    config: {max_steps: 400}     # robot env fields
    pose: {position: [0.0, 0.0, 0.0]}
    policy: scripted             # optional per instance; "idle" if omitted
viewer: {mode: none}             # none | native | web | both
```

Validation resolves every `robot`/`task`/`policy` name through the registries
and checks scene/robot-kind/task compatibility without constructing a robot or
model. `ros2_real` is accepted for forward compatibility but raises "not yet
implemented". `create_session()` supports `backend: direct`: one robot becomes
a `create_runtime()` composition, a `world` becomes a `SharedWorld` (whose
robots run no task or policy yet). It injects the `simulation` seed and
randomization into any robot config that declares those fields and rejects a
robot config that repeats them.

`scripts/run_sim.py` builds every run this way. Its older `--config` (task
file), `--world` (world file), and bare `--robot` inputs are converted by
`physai.config.compat` and print a deprecation notice; the loaders in
`physai.config.legacy` and the files under `configs/tasks/` and
`configs/worlds/` remain for that window and for `scripts/run_ros2_sim.py`.
Worked examples live in `configs/manifests/`. The decision is recorded in
[ADR 10](adr/0010-manifest-adoption.md).

## Host + client API

`physai.web.host.Host` is the one host class. `Host.for_robot(...)` builds a
direct-MuJoCo session (one `RobotPort`, optional policy) and
`Host.for_world(...)` a `SharedWorld` session (N namespaced instances,
command/hold only, no policy); a single robot is a `Host` with one instance.
Every method is instance-keyed with `instance_id` optional, defaulting to the
sole instance:

```text
Host
 ├── scene() / list_robots()               capability + geometry discovery
 ├── latest_state()                        dynamic transform snapshot
 ├── camera_jpeg(name, instance_id=None)   cached camera JPEG
 ├── submit(action, instance_id=None, ...) command + control lease
 ├── reset() / set_paused(paused)          world-atomic, affects every client
 ├── start_recording() / stop_recording(success)   browser episode capture
 │   / recording_status()                    (single-instance, needs record_dir)
 ├── list_episodes() / load_episode(file)   replay a saved episode on the paused
 │   / seek(frame, relative) / set_playback(playing, speed) / exit_playback()
 ├── release_control(source)               drop every instance a source owns
 └── start() / stop()                      physics-thread lifecycle
```

One physics thread runs the control loop (30 Hz by default). A separate camera
worker renders named cameras on its own cadence (`CameraFeed.PERIOD`,
1/30 s), so capture never pauses physics. `Host.physics_lock` serializes
MuJoCo access between the two threads and renderer clients such as the native
`--viewer`; HTTP camera requests only read the latest cached JPEG.

Reset and pause are world-atomic and affect every client. Manual control uses
a short per-instance lease: the first client to submit becomes the controller
and other clients are rejected while it is alive; disconnecting or letting it
expire returns the instance to its hold action or policy.
`release_control(source)` drops only the instances `source` owns.

`web/app.py` (FastAPI) is a thin client of `Host`: it never branches on host
type and never imports `mujoco` or a robot module. The surface is the same for
single- and multi-robot sessions:

| Route | Behavior |
| --- | --- |
| `GET /` | Three.js browser console (static file) |
| `GET /api/robots` | Every instance's id, kind, action modes, capabilities, and cameras |
| `GET /api/scene` | Static geometry manifest for the whole session |
| `GET /api/episodes` | Saved episodes of the record directory (file, length, `success`, `playable`); `[]` when recording is disabled |
| `GET /api/state` | Latest transform snapshot for the whole session |
| `GET /api/mesh/{id}` | Compiled mesh binary payload |
| `GET /api/camera/{name}.jpg` | Cached camera JPEG (single snapshot), optionally `?robot=<instance_id>` |
| `GET /api/camera/{name}/stream` | MJPEG `multipart/x-mixed-replace` stream; 404 on an unknown camera |
| `WS /ws` | Accepts `select_robot`, `command`, `reset`, `pause`, `release_control`, `record_start`, `record_stop`, `playback_load`/`_seek`/`_step`/`_play`/`_exit`; streams state + errors |

The streamed state carries transforms for the whole session plus:
`gripper_contacts` (each with `force_n`, the pad's summed contact normal force
in newtons, `null` during playback); `ee_pose` (a `PoseStamped`-shaped dict
with `reference` `tool` or `ee_pose`, or `null`; `tool` is the pose from the
robot kinematics' optional `tool_pose(data)`, the pinch centre for SO-101); and
the `paused`, `recording`, and `playback` status blocks, merged in at read
time by `Host.latest_state()` so a paused world still reports changes.
Recording and playback are specified in
[ADR 9](adr/0009-web-session-recording-and-playback.md).

Both clients depend only on this API plus `RobotSpec` capabilities:

- **`--viewer` (native MuJoCo):** `mujoco.viewer.launch_passive` renders a
  private `MjData` copy refreshed under `Host.physics_lock` (see
  [ADR 4](adr/0004-tk-viewer-frozen.md)). It has no camera panel of its own;
  combine it with `--serve`.
- **`--serve` (FastAPI + Three.js):** the sophisticated client; all new UI
  features go here. See [docs/WEB_VIEWER_RUNBOOK.md](WEB_VIEWER_RUNBOOK.md)
  for keyboard mapping, cloud-workspace setup, and troubleshooting.

Shared-world instances are built through the robot registry
(`robots.registry.create_shared_instance()`); `web/host.py` and `sim/world.py`
never branch on a robot name, and `SharedWorld` takes an optional
`shared_attach` hook so a robot can inject shared-world-only MJCF (for example
SO-101's extra cameras).

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

```text
so101 + pick_place + scripted, visual-servo, or Planner + PlanRunner + MuJoCo
turtlebot4 + generic smoke test + constant twist policy + MuJoCo
```

`so101` and `turtlebot4` are registered embodiments, not names that belong in
generic contracts. `pick_place` needs arm and gripper capabilities, so the
policy, demo, and planner workflows are SO-101-specific today. TurtleBot4
(native model, differential drive, RPP navigation, ROS2/Nav2 acceptance path)
shows the capability abstraction generalizes and is not a development focus
([ADR 5](adr/0005-turtlebot4-kept-as-second-embodiment.md)). The pick-place
and sorting tasks are minimal baselines (`physai.tasks.pick_place_minimal`,
`sorting_minimal`; registry keys `pick_place` and `sorting`).

## Model roles

No model-backed planner is built in core: `physai.planner` defines the
`Planner` contract and a scripted backend, and a VLM-grounded backend belongs
in `research/vlm_planners/`. ACT is a low-level policy checkpoint format behind
the policy boundary (`LeRobotPolicy`,
`research/imitation_learning/vla_adapter.py`); it predicts control-rate
actions, is not a planner, and is never coupled to a robot environment.
Checkpoints are written by `train_act.py` into the ignored `outputs/`
directory and loaded through an explicit path.

## Demonstration data

Demonstrations are LeRobot-shaped arrays. Feature names, camera streams,
action layout, and encoder belong to the robot's `RobotTrainingContract`
(`src/physai/robots/<robot>/contracts.py`). For SO-101:

```text
observation.images.front  (T, H, W, 3) uint8
observation.images.wrist  (T, H, W, 3) uint8
observation.state         (T, 6) float32, radians
action                    (T, 6) float32, absolute joint targets
```

The six values are `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
wrist_roll, gripper`. Recording and loading belong to `physai.data`, which
consumes the robot contract without importing a concrete robot. The canonical
schemas are `so101_observation_spec()`/`so101_action_spec()` (TurtleBot4 has
twist and wheel-state equivalents), and dataset metadata serializes them
rather than defining a second layout. The recorder writes a compact `.npz`
with LeRobot-shaped keys; a future `LeRobotDataset` exporter must reuse the
same metadata.

A recording may add `observation.environment_state` (`(T, nq)` float64, the
full simulator qpos), declared in `meta.json`'s `features`; the web recorder
and `scripts/collect_demos.py` write it (see
[ADR 9](adr/0009-web-session-recording-and-playback.md)). Metadata also
records the scene name and configuration snapshot, so runs stay reproducible
as the scene registry grows.

## ROS2 boundary

Direct MuJoCo runs in one process without ROS2. `src/physai/contracts.py`
mirrors the ROS2 types while staying transport-neutral, and
`src/physai/bridge/ros2_contract.py` is the source of truth for topics,
message types, rates, and frames (it is not itself an adapter). The
synchronous bridge core and the real `rclpy` nodes target ROS2 Jazzy on
Ubuntu 24.04.

Two interchangeable adapter roles sit behind it: `ROS2MuJoCoAdapter` (ROS2
topics -> MuJoCo control, publishing simulated state and camera frames) and
`ROS2HardwareAdapter` (the same topics -> the embodiment's motor and camera
interfaces). Each subscribes to the joint trajectory and gripper endpoints,
decodes them into the shared `Action` contract, exposes the latest complete
command through a synchronous tick API, and publishes canonical state through
an injected `MessageCodec` (`ContractMessageCodec` for transport-neutral
tests, `ROS2MessageCodec` for real messages without making `rclpy` a core
dependency). Both must make unit conversion, joint order, timestamps, frame
names, command freshness, and rate explicit; joint order and shape are
validated at decode time against `RobotSpec`.

`rclpy` is imported lazily inside each robot's `ros2_node.py`, which is how
`physai.sim` stays free of it (enforced by import-linter) even though
`robots.registry` imports those modules.

Before hardware deployment, pass in order: (1) direct-MuJoCo contract tests;
(2) ROS2 MuJoCo integration tests on the hardware driver's topics and types;
(3) hardware-driver tests with recorded or fake joint states and frames;
(4) a supervised hardware smoke test with command timeout, joint limits,
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

- **Jog resolver construction.** A robot registers its jog resolver on
  `RobotDescriptor.jog` and the Host asks the registry
  (`robots.registry.create_jog_resolver()`), so the Host no longer probes the
  robot. The jog math is shared (`robots/so101/jog.py`), but
  `TwistToJointResolver` is still built separately in `robots/so101/env.py`,
  `robots/so101/shared.py`, and `scripts/teleop_keyboard.py`.
- **Host mode branching.** `Host` still branches on single-robot vs shared
  world in about a dozen places, because only a single robot runs a policy,
  records, or plays back. The branches go away when shared-world sessions gain
  tasks and policies (see below), not by splitting `Host` in two; its lease
  (`web/lease.py`) and camera worker (`web/cameras.py`) are already separate.
- **`scripts/` composition-root adoption.** Scripts other than `run_sim.py`
  (`workspace_map.py`, `benchmark_ik.py`, `render_docs_media.py`,
  `teleop_keyboard.py`, `collect_demos.py`, `eval_policy.py`,
  `eval_randomization.py`, `plan_task.py`) still hand-assemble `EnvConfig`/env
  objects instead of calling `create_runtime()` or `create_session()`. They
  are safety-gated either way (the gate lives in the adapter, not the
  composition path).
- **Legacy configuration.** `--config`, `--world`, `configs/tasks/`, and
  `configs/worlds/` are deprecated in favor of manifests and are removed after
  a deprecation window; `scripts/run_ros2_sim.py --config` still reads the
  task file. Shared-world sessions run no tasks or policies yet.
- **Registry granularity.** Adding a robot is "one `RobotDescriptor` + one
  `register_embodiment()` call," not literally one line — a robot supplying
  every optional factory sets up to six fields on that one descriptor.
