# Architecture

The internal design reference: the frozen design shape, module ownership, dependency
rules, stable contracts, the session manifest, the host and client API, extension seams
and the research/core boundary. Setup and runnable commands are in
[README.md](../README.md), agent workflow rules in [AGENTS.md](../AGENTS.md), and the
decisions behind the frozen shape in [docs/adr/](adr/README.md).

## Design shape

The stack separates embodiment, task and decision-making so each can evolve
independently. A runtime composition selects one or more robots, one task where
applicable, and one planner or policy per session, then connects them through
message-shaped contracts.

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

The runnable baselines are the scripted planner/policy path with `PlanRunner` and the
SO-101 visual-servo policy; model-backed planners and VLA policies plug into the same
boundaries from `research/`.

## Execution modes and deployment parity

A simulator simulates; ROS2 transports. They are different layers, and three paths use
them:

```text
Fast development:        Planner/Policy -> direct RobotPort -> simulator
Integration validation:  Planner/Policy node -> ROS2 topics -> ROS2SimAdapter -> simulator
Real deployment:         Planner/Policy node -> ROS2 topics -> ROS2HardwareAdapter -> hardware
```

The two ROS2 paths expose the same topic, message, unit, frame, joint-order and rate
contracts, so a policy node moves from simulation to hardware unchanged. Direct mode is
the fast path for unit tests, training and regression; it does not replace ROS2
integration validation.

`RobotPort` has three transport adapters named after the manifest's `backend` field:
`DirectAdapter` (`direct`) and `ROS2SimAdapter` (`ros2_sim`) are implemented;
`ROS2HardwareAdapter` (`ros2_real`) is a contract-only placeholder (simulation only for
now, see [ROADMAP.md](../ROADMAP.md)). None is simulator-specific: each wraps whichever
`RobotPort` a robot factory built, so transport (direct or ROS2) and simulator engine
(MuJoCo or Isaac) are independent axes. Transports are registry-driven
(`robots.adapters.register_adapter()` / `create_adapter()`). The direct adapter keeps a
synchronous `reset`/`step`; the ROS2 adapters may use callbacks and queues internally but
translate to the same `Observation` and `Action` at the boundary.

| Mode | Primary value | Residual risk |
| --- | --- | --- |
| Direct | Fast, deterministic iteration and CI | Does not exercise ROS2 timing, QoS, TF or serialization |
| Simulator through ROS2 | Tests the production message and process boundary before hardware | Adds setup, scheduling, latency and transport failure modes |
| Hardware through ROS2 | Tests real sensors, actuators, calibration and safety | Slow, nondeterministic, hardware-dependent, higher risk |

## Module ownership

| Module | Owns | Must not own |
| --- | --- | --- |
| `physai.contracts` | Shared `Observation`, `Action` and ROS2-shaped value types | Robot-specific ordering or task rules |
| `physai.config` | Typed YAML parsing: `manifest` (the session manifest) and `legacy` (`load_sim_config`, `load_task_config`, kept for the deprecated CLI inputs) | Simulation behavior, robot construction or task evaluation |
| `physai.robots` | Embodiment discovery, `RobotSpec`, robot ports, factories, environments, adapters, the backend registry, shared-world instance/attach factories | Task reward, planner decisions or model SDKs |
| `physai.tasks` | Task state, reset rules, reward, metrics, termination | Robot internals or action generation |
| `physai.sim` | One subpackage per engine plus the neutral modules both read (`studio`: the common look; `workspace`: table, target, cubes and front camera) | Engine code in the namespace; importing `physai.sim` must never need an engine SDK, and the neutral modules never import one |
| `physai.sim.mujoco` | MuJoCo core, scene builders, the shared multi-robot world, rendering, simulation time | Robot-specific environment logic, ROS2 transport, or any robot-name branch (`SharedWorld` takes an optional `shared_attach` hook) |
| `physai.sim.isaac` | `SimulationApp` lifecycle, applying a `RobotDescription` to a USD stage, world extras (lighting, ground plane), the workspace builder `add_workspace` | Robot-specific environment logic or task composition (`robots.so101.isaac_env` is the only other module that may import Isaac); importing `physai.sim.mujoco` |
| `physai.planner` | Instruction and image grounding, `Plan`, `SubGoal`, the planner registry | Control-rate motor commands; research planners |
| `physai.policy` | Control-rate `Action` production, core baselines (`constant`, `constant_twist`, `replay`), the policy registry | Task scoring, robot discovery, checkpoint-backed inference |
| `physai.control` | Action resolution, capability checks, rate limiting | High-level planning or task semantics |
| `physai.data` | Episode recording, dataset and checkpoint metadata, evaluation, the Gymnasium adapter | Simulation decisions or model inference |
| `physai.bridge` | ROS2 transport, topic and message mapping, timing, ROS2-backed adapters | Physics, task semantics or model inference |
| `physai.runtime` | Runtime composition, compatibility checks, safety orchestration | Robot-specific physics, task reward or model inference |
| `physai.web` | The one `Host`, the FastAPI client surface, telemetry, the static browser client | Simulator selection or robot construction |
| `research/<topic>/` | Approach-specific implementations that self-register on import | Anything a core module imports ([Research boundary](#research-boundary)) |
| `scripts/` | CLI argument parsing and runtime composition | IK, reward calculation or SDK-specific code |
| `launch/` | Generic ROS2 launch composition | Robot physics, transport callbacks or task rules |
| `tests/` | Executable behavior and contract coverage | New runtime ownership |

```text
src/physai/
├── contracts.py       shared message-shaped values
├── config/            typed YAML configuration (manifest.py; legacy.py and compat.py are deprecated)
├── robots/            embodiment ports, adapters, registries, factories
│   ├── description.py sim-neutral RobotDescription schema + YAML loader
│   ├── so101/         environments (mujoco_env.py, isaac_env.py), layouts, kinematics, shared-world adapter
│   └── turtlebot/     environment and shared-world adapter
├── tasks/             task rules and registry
├── sim/               one subpackage per engine, plus neutral studio.py and workspace.py
│   ├── mujoco/        core, shared world, domain randomization, scenes/
│   └── isaac/         core, RobotDescription -> USD, world extras (optional backend)
├── planner/  policy/  control/  data/  bridge/  runtime/  web/

research/
├── scripted_experts/   privileged-ground-truth demo-collection policies
├── classical_control/  visual servo and other model-free baselines
├── imitation_learning/ ACT/LeRobot dataset tooling, training, checkpoints
└── vlm_planners/       model- or heuristic-grounded Planner implementations
```

## Dependency direction

Enforced by `uv run lint-imports` (`pyproject.toml`'s `[tool.importlinter]`) and run
inside the normal suite via `tests/core/boundaries/test_import_boundaries.py`, so a
violation fails `pytest tests/ -q`.

| Rule | Rationale |
| --- | --- |
| `physai.sim` (and `mujoco`/`isaac`) must not import `physai.bridge` or `rclpy` | Keeps ROS2 optional for simulator-only workflows |
| `physai.policy`, `physai.tasks`, `physai.planner` must not import `mujoco` | They depend on ports, never MuJoCo types. One accepted indirect path: `policy.registry` calls `robots.registry.create_robot_policy()` to build a robot-owned policy without importing the robot |
| `physai` must not import `research` | Research registers itself into a core registry on import |
| `physai.robots.description` and `physai.sim.workspace` must not import `mujoco` or Isaac | The sim-neutral descriptions are read by every simulator's builder ([ADR 11](adr/contracts-and-descriptions.md#adr-11-sim-neutral-robot-description), [ADR 17](adr/contracts-and-descriptions.md#adr-17-sim-neutral-workspace-description)) |
| Only `physai.sim.isaac` and `physai.robots.so101.isaac_env` may import `isaacsim`, `omni`, `pxr` | Isaac is an optional, GPU-only backend (`uv sync --extra isaac`); everything else reaches it through the `RobotPort`/`RobotSpec` ports |
| `physai.sim.mujoco` and `physai.sim.isaac` must not import each other, and `physai.sim.isaac` must not import `mujoco` | Peer backends ([ADR 14](adr/simulators.md#adr-14-simulator-package-layout-and-adapter-names)) |
| A robot-name branch in generic code is a bug | `SharedWorld`, `web.host.Host` and the web client resolve behavior through `RobotSpec` capabilities or a registry lookup. A code-shape rule enforced by review, not by import-linter |

## Design patterns in use

- **Ports and Adapters:** `Observation`, `Action`, `RobotPort`, `Planner`, `Policy` and
  `Task` are ports; direct simulators, ROS2 transport, model SDKs and concrete robots are
  adapters around them. Planners, policies and tasks are interchangeable behind their
  contracts (Strategy).
- **Registry plus Factory:** robot, scene, task, policy, planner and backend registries
  map stable names to factories. `register_embodiment()` registers everything one robot
  owns from one `RobotDescriptor`. Builtins load lazily to keep optional dependencies out
  of model-free workflows; a research module registers itself on import instead.
- **Composition Root:** `physai.runtime.create_runtime()` (and `create_session()` for a
  manifest) is the intended composition root; scripts should only parse arguments and call
  it. `run_sim.py` does; the others do not yet ([Known remaining gaps](#known-remaining-gaps)).
- **Safety Gate:** `SafetyController` validates action mode, joint order, finite values,
  timestamps, joint limits and per-joint step limits immediately before a robot port
  receives a command. It lives inside the adapters (`DirectAdapter`, `MuJoCoROSBridge`, the
  Gymnasium adapter), so no workflow reaches the robot unchecked. A refused action raises
  `SafetyViolation` (a `ValueError`), which an evaluation records as an unsafe action
  instead of aborting. `RobotSpec.max_joint_delta` bounds a command against the *measured*
  position and must stay above `JointRateLimiter`'s command-to-command clamp, or normal
  servo lag trips the gate.

These patterns are deliberately lightweight: add an abstraction only when it removes
coupling at a boundary or makes a component replaceable.

## Extension seams

Every seam is one new file plus one registration call, except where noted.

| Seam | New file | Registration |
| --- | --- | --- |
| Robot | `robots/<name>/` package | One `RobotDescriptor` + `register_embodiment(name, descriptor)` in `robots/registry.py:_load_builtins()` (a robot with every optional factory sets up to six fields on that one descriptor) |
| Scene | `sim/mujoco/scenes/<name>.py` | `register_scene(name, SceneDefinition(...))` in `sim/mujoco/scenes/registry.py:_load_builtins()` |
| Task | `tasks/<name>.py` | `register_task(name, factory)` in `tasks/registry.py:_load_builtins()` |
| Policy (core baseline) | `policy/<name>.py` | `register_policy(name, factory)` in `policy/registry.py:_load_builtins()` |
| Policy (robot-owned or research) | `robots/<robot>/<name>.py` or `research/<topic>/<name>.py` | `register_robot_policy(robot_name, policy_name, factory)`, or `register_policy(name, factory)` for a global one; a research module calls it on import |
| Planner (core baseline) | `planner/base.py` | `register_planner(name, factory)` in `planner/registry.py:_load_builtins()` |
| Planner (research) | `research/<topic>/<name>.py` | `register_planner(name, factory)` on import |
| Transport adapter | `robots/adapters.py` (a builder function) | `register_adapter(name, builder)` in `robots/adapters.py:_load_builtins()` |
| Simulator engine | `sim/<engine>/` package (mirrors `sim/mujoco/`) | Add the engine to a `RobotDescriptor.simulators` tuple in `robots/registry.py:_load_builtins()`; the robot's factory branches on `simulator=` ([ADR 15](adr/simulators.md#adr-15-simulator-engine-selection)) |
| Client capability control | `web/static/js/controls.js` / `ui.js` | None: capabilities are declarative data (`RobotSpec.action_modes`/`capabilities`) rendered conditionally |

## Frozen vs. free to change

| Layer | Frozen (needs an ADR + version bump) | Free to change |
| --- | --- | --- |
| Contracts | `contracts.py` (`Observation`, `Action`, `*Spec`, `Header`, `JointState`, `Pose`, `Twist`, ...) | Internal helpers outside the dataclass shape |
| Ports | `robots/base.py` (`RobotPort`, `RobotSpec`, `RobotTrainingContract`, `KinematicsPort`); `policy/base.py`; `tasks/base.py`; `planner/base.py` (`Planner`, `Plan`/`SubGoal`) | Concrete implementations |
| Manifest schema | `SessionManifest`'s field names, types and rules once a session depends on them | Which YAML files exist under `configs/` |
| Host API | The method surface in [Host + client API](#host--client-api) | `Host`'s threading model, camera cadence, lease timeout |
| Folder tree | Top-level package boundaries, `sim/`'s one-subpackage-per-engine shape, and core never importing `research/` | File layout inside one robot, research topic or simulator backend |
| Registries | The register/create pattern | Which factories are registered, and in what order |

Evolution rule: a frozen item changes only via (1) an ADR, (2) a version marker on the
affected contract (for example `SessionManifest.schema_version`) and (3) a deprecation
window: old names and routes keep working with a warning for at least one minor version.

## Stable contracts

`Observation` carries joint state, named image frames, an optional end-effector pose, step
count and simulation time. `Action` carries either joint targets or a Cartesian/base
`Twist`, plus a normalized gripper command; the robot's capabilities decide which is valid,
and an action never carries both modes.

Camera resolution is chosen in one place: the manifest's `simulation.camera_resolution` or
a script's `--camera-res`, one of `physai.contracts.CAMERA_RESOLUTIONS` (320x240 default,
640x480, 1280x720). Both simulators' scene configs, the Isaac env, the web host, the
training contract and the robot description's camera entries read it; no scene, env or test
sets its own size, so results from different simulators or runs see the same image.

`RobotSpec` describes an embodiment without exposing simulator or hardware API: joint
names, action modes, observation modalities, named capabilities, joint limits and optional
per-joint command-step limits. Workflow code calls `supports()`/`require()` instead of
branching on a robot name. Registration also declares an embodiment kind (`robot_kind()`)
and robot-owned `scene_defaults()`; scene definitions declare compatible robot kinds and
task names, so an incompatible combination fails during composition rather than inside a
policy or scene builder.

Two ports surround these contracts:
- `RobotPort` owns observation acquisition, action dispatch, lifecycle and the
  robot-specific mapping of joint names, units, gripper range, cameras and frames.
- `KinematicsPort` owns FK, IK, Jacobian and pinch-frame operations. A MuJoCo
  implementation may use `MjModel`/`MjData`; a hardware implementation must use measured
  joint state and a calibrated model. Kinematics are embodiment-specific: the SO-101's
  (`robots/so101/kinematics.py`) is not a generic service, and tasks and policies request
  capabilities (`joint_position`, `arm_kinematics`, `base_velocity`, `odometry`), never
  robot names.

`Policy`, `PlanRunner` and task code depend on these ports, never on simulator types.
`SO101Env` is a robot-owned backend (observation, action, lifecycle, embodiment state) and
creates no task; `TaskRuntime` wraps a robot port with a registered task and owns reset,
metrics, reward, success hold and termination. `Planner` maps an instruction and
observation to a `Plan` of `SubGoal`s and optional `PoseStamped` waypoints; `Policy` maps an
observation and optional goal to one `Action` per control tick.
`physai.policy.replay.VLAPolicy` (chunked-action shape: observation packing, buffering,
unit decoding) stays in core because the model-free `ReplayPolicy` needs it;
checkpoint-backed subclasses (`LeRobotPolicy`) are research.

`physai.runtime.create_runtime` validates the task's required capabilities against the
`RobotSpec`, resolves the scene, wraps the robot with `TaskRuntime` when a task is selected,
and creates an optional registered policy. Safety is not its job: the adapter gates every
action. `create_runtime(scene_name=...)` rejects task-scene mismatches during composition,
and a scene's embodiment defaults come from the robot's `scene_defaults()`, so the scene
layer never looks up a robot by name.

Registered compositions today: `so101` + `single_cube_fixed_place` (or `sorting`) with the
scripted, visual-servo or Planner + `PlanRunner` policies, and `turtlebot4` with a constant
twist policy. `single_cube_fixed_place` needs arm and gripper capabilities, so the policy,
demo and planner workflows are SO-101-specific; TurtleBot4 shows the capability abstraction
generalizes and is not a development focus
([ADR 5](adr/repo-scope.md#adr-5-turtlebot4-stays-as-the-second-embodiment)).

### Task-specific scenes

```text
sim/workspace.py (neutral: imports neither engine)
     +-- WorkspaceConfig: table, target, front camera, cubes()
     +-- shared friction, target colour and camera field-of-view constants

sim/mujoco/scenes/common.py
     +-- WorldSceneConfig: WorkspaceConfig plus the MuJoCo-only model settings
     +-- ManipulationSceneConfig: configurable end-effector and pad attachments
     +-- shared manipulation-world builder (reads WorkspaceConfig)
sim/mujoco/scenes/single_cube_fixed_place.py   one cube and one target layout
sim/mujoco/scenes/sorting_minimal.py           colored cube layout and sorting positions

sim/isaac/scene.py
     +-- add_workspace: the same WorkspaceConfig as USD prims
```

Each scene describes its cubes through `cubes()`, builds MuJoCo through
`build_spec()`/`build_model()`, and names its object arrangement through a class-level
`layout_kind`; the robot maps that name to a layout strategy (`robots/so101/layout.py`)
that places the objects each episode and reads them back, so a new arrangement adds a
layout rather than editing the environment. Robot model paths, end-effector anchors and
the gripper's pad and wrist-camera fit are configuration the robot supplies through
`scene_defaults()`. Selecting a scene is selecting a class. See
[ADR 17](adr/contracts-and-descriptions.md#adr-17-sim-neutral-workspace-description).

## Session manifest

One YAML file describes a run: robot instances, scene, task, policy, backend, simulation
settings and viewer options. A single-robot run is a manifest with one entry in `robots`;
a `world` block (or several robots) makes one shared MuJoCo world.
`physai.config.manifest.load_manifest()` validates it (the field list and rules are in that
module's docstring) and `physai.runtime.create_session()` builds it:

```yaml
schema_version: 1
simulation: {seed: 0}            # the one source of seed and randomization
scene:
  name: single_cube_fixed_place
  overrides: {target_radius: 0.04}  # scene fields, including robot_xml
simulator: mujoco                # optional; default mujoco
backend: direct                  # direct | ros2_sim | ros2_real
task: single_cube_fixed_place    # optional session-wide default
success_hold_steps: 10
robots:
  - id: arm_1
    robot: so101
    config: {max_steps: 400}     # robot env fields
    pose: {position: [0.0, 0.0, 0.0]}
    policy: scripted             # optional per instance; "idle" if omitted
viewer: {mode: none}             # none | native | web | both
```

Validation resolves every `robot`, `task` and `policy` name through the registries and
checks scene, robot-kind and task compatibility without constructing a robot or model.
`ros2_real` is accepted for forward compatibility but raises "not yet implemented".
`create_session()` supports `backend: direct`: one robot becomes a `create_runtime()`
composition, a `world` becomes a `SharedWorld` (whose robots run no task or policy yet). It
injects the `simulation` seed and randomization into any robot config that declares those
fields and rejects a robot config that repeats them.

`simulator` picks the physics engine for every robot in the session
(`RobotDescriptor.simulators` declares what a robot supports; only so101 has more than
one). A non-MuJoCo engine rejects a `world` block, several robots, `backend: ros2_sim` and
any `viewer.mode` other than `none`
([ADR 15](adr/simulators.md#adr-15-simulator-engine-selection);
`configs/manifests/so101_single_cube_fixed_place.yaml` runs on both engines). `run_sim.py --sim {mujoco,isaac}`
overrides it through `physai.config.compat.with_overrides`, which re-runs the validation.

`run_sim.py` builds every run this way. Its older `--config` (task file), `--world` and bare
`--robot` inputs are converted by `physai.config.compat` with a deprecation notice; those
loaders (`physai.config.legacy`) and `configs/tasks/`, `configs/worlds/` remain for that
window and for `scripts/run_ros2_sim.py --config`. Worked examples are in
`configs/manifests/`; the decision is
[ADR 10](adr/config-and-sessions.md#adr-10-the-manifest-becomes-the-run-description).

## Host + client API

`physai.web.host.Host` is the one host class. `Host.for_robot(...)` builds a direct-MuJoCo
session (one `RobotPort`, optional policy) and `Host.for_world(...)` a `SharedWorld` session
(N namespaced instances, command and hold only, no policy); a single robot is a `Host` with
one instance. Every method is instance-keyed with `instance_id` optional, defaulting to the
sole instance:

```text
Host
 ├── scene() / list_robots()               capability + geometry discovery
 ├── latest_state()                        dynamic transform snapshot
 ├── camera_jpeg(name, instance_id=None)   cached camera JPEG
 ├── submit(action, instance_id=None, ...) command + control lease
 ├── reset() / set_paused(paused)          world-atomic, affects every client
 ├── start_recording() / stop_recording(success) / recording_status()   (single-instance, needs record_dir)
 ├── list_episodes() / load_episode(file) / seek(frame, relative) / set_playback(playing, speed) / exit_playback()
 ├── release_control(source)               drop every instance a source owns
 └── start() / stop()                      physics-thread lifecycle
```

One physics thread runs the control loop (30 Hz by default). A separate camera worker
renders named cameras on its own cadence (`CameraFeed.PERIOD`, 1/30 s), so capture never
pauses physics. `Host.physics_lock` serializes MuJoCo access between the two threads and
renderer clients such as the native `--viewer`; HTTP camera requests only read the latest
cached JPEG. Reset and pause are world-atomic. Manual control uses a short per-instance
lease: the first client to submit becomes the controller and others are rejected while it
is alive; disconnecting or letting it expire returns the instance to its hold action or
policy.

`web/app.py` (FastAPI) is a thin client of `Host`: it never branches on host type and never
imports `mujoco` or a robot module. The surface is the same for single- and multi-robot
sessions:

| Route | Behavior |
| --- | --- |
| `GET /` | Three.js browser console (static file) |
| `GET /api/robots` | Every instance's id, kind, simulator, action modes, capabilities and cameras |
| `GET /api/scene` | Static geometry manifest for the whole session |
| `GET /api/episodes` | Saved episodes of the record directory (file, length, `success`, `playable`); `[]` when recording is disabled |
| `GET /api/state` | Latest transform snapshot for the whole session |
| `GET /api/mesh/{id}` | Compiled mesh binary payload |
| `GET /api/camera/{name}.jpg` | Cached camera JPEG, optionally `?robot=<instance_id>` |
| `GET /api/camera/{name}/stream` | MJPEG `multipart/x-mixed-replace` stream; 404 on an unknown camera |
| `WS /ws` | Accepts `select_robot`, `command`, `reset`, `pause`, `release_control`, `record_start`, `record_stop`, `playback_load`/`_seek`/`_step`/`_play`/`_exit`; streams state and errors |

The streamed state carries transforms for the whole session plus `gripper_contacts` (each
with `force_n`, the pad's summed contact normal force in newtons, `null` during playback),
`ee_pose` (a `PoseStamped`-shaped dict with `reference` `tool` or `ee_pose`, or `null`;
`tool` is the pose from the kinematics' optional `tool_pose(data)`, the pinch centre for
SO-101) and the `paused`, `recording` and `playback` blocks, merged at read time by
`Host.latest_state()`. Recording and playback are specified in
[ADR 9](adr/web-host.md#adr-9-record-and-replay-web-sessions-through-the-existing-data-path).

Both clients depend only on this API plus `RobotSpec` capabilities. `--viewer` (native
MuJoCo,
[ADR 4](adr/web-host.md#adr-4---viewer-is-mujocos-own-viewer-frozen-in-scope)) has no
camera panel of its own; `--serve` (FastAPI + Three.js) is where all new UI features go
([runbook](WEB_VIEWER_RUNBOOK.md)). Shared-world instances are built through
`robots.registry.create_shared_instance()`; `web/host.py` and `sim/mujoco/world.py` never
branch on a robot name, and `SharedWorld` takes an optional `shared_attach` hook so a robot
can inject shared-world-only MJCF (for example SO-101's extra cameras). Jogging is a
capability a robot registers (`RobotDescriptor.jog`) and the host asks the registry for
(`create_jog_resolver()`); the web client hides controls a robot's capabilities do not
support.

## Research boundary

Research plugs in through the existing contracts only (`Planner`, `Policy`, `Task`,
`Observation`/`Action`, `RobotTrainingContract`, `physai.data`, the Gymnasium adapter) and
registers itself into a core registry on import. Core ships the contracts plus minimal
baselines:

| Module | Home | Why |
| --- | --- | --- |
| `ScriptedPlanner`, `PlanRunner`, `ReplayPolicy`/`VLAPolicy`, `SortingTask`, `GymnasiumAdapter`, TurtleBot4 `navigation.py` | Core | Minimal, technique-agnostic baselines |
| `research/scripted_experts/so101_pick_place_expert.py` | Research | Privileged expert (reads `mujoco` object pose directly) |
| `research/classical_control/so101_visual_servo.py` | Research | One calibrated-camera baseline, not infrastructure |
| `research/imitation_learning/{act_dataset,vla_adapter,train_act}.py` | Research | ACT/LeRobot pipeline and the checkpoint-backed `LeRobotPolicy` |
| `research/vlm_planners/sorting_planner.py` | Research | Task-coupled (reads privileged `env.cube_positions`) |

See [ADR 3](adr/repo-scope.md#adr-3-research-code-lives-outside-the-core-package) and
[research/README.md](../research/README.md). No model-backed planner is built in core: a
VLM-grounded backend belongs in `research/vlm_planners/`. ACT is a low-level policy
checkpoint format behind the policy boundary (`LeRobotPolicy`); it predicts control-rate
actions, is not a planner, and is never coupled to a robot environment. Checkpoints are
written by `train_act.py` into the ignored `outputs/` and loaded through an explicit path.

The boundary also holds at test time: `tests/core/` (mirroring `physai`'s subpackages) sits
next to `tests/research/<topic>/`, each in its own CI job (`test-core`, `test-research`), so
a research topic's failures and heavier dependencies (such as torch and LeRobot in the `training` extra) stay
isolated.

## Demonstration data

Demonstrations are LeRobot-shaped arrays. Feature names, camera streams, action layout and
encoder belong to the robot's `RobotTrainingContract`
(`src/physai/robots/<robot>/contracts.py`). For SO-101:

```text
observation.images.front  (T, H, W, 3) uint8
observation.images.wrist  (T, H, W, 3) uint8
observation.state         (T, 6) float32, radians
action                    (T, 6) float32, absolute joint targets
```

The six values are `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll,
gripper`. Recording and loading belong to `physai.data`, which consumes the robot contract
without importing a concrete robot. The canonical schemas are `so101_observation_spec()` and
`so101_action_spec()` (TurtleBot4 has twist and wheel-state equivalents), and dataset
metadata serializes them rather than defining a second layout. The recorder writes a compact
`.npz` with LeRobot-shaped keys; a future `LeRobotDataset` exporter must reuse the same
metadata. A recording may add `observation.environment_state` (`(T, nq)` float64, the full
simulator `qpos`), declared in `meta.json`'s `features`; the web recorder and
`scripts/collect_demos.py` write it
([ADR 9](adr/web-host.md#adr-9-record-and-replay-web-sessions-through-the-existing-data-path)).
Metadata also records the scene name and configuration snapshot, so runs stay reproducible.

## ROS2 boundary

Direct mode runs in one process without ROS2. `src/physai/contracts.py` mirrors the ROS2
types while staying transport-neutral, and `src/physai/bridge/ros2_contract.py` is the source
of truth for topics, message types, rates and frames (it is not itself an adapter). The
synchronous bridge core and the `rclpy` nodes target ROS2 Jazzy on Ubuntu 24.04.

Two interchangeable adapter roles sit behind it: `ROS2SimAdapter` (ROS2 topics to simulator
control, publishing simulated state and camera frames) and `ROS2HardwareAdapter` (the same
topics to the embodiment's motor and camera interfaces). Each subscribes to the joint
trajectory and gripper endpoints, decodes them into the shared `Action`, exposes the latest
complete command through a synchronous tick API, and publishes canonical state through an
injected `MessageCodec` (`ContractMessageCodec` for transport-neutral tests, `ROS2MessageCodec`
for real messages without making `rclpy` a core dependency). Both make unit conversion, joint
order, timestamps, frame names, command freshness and rate explicit; order and shape are
validated at decode time against `RobotSpec`. `rclpy` is imported lazily inside each robot's
`ros2_node.py`, which keeps `physai.sim` free of it.

Before hardware deployment, pass in order: (1) direct contract tests; (2) ROS2 simulator
integration tests on the hardware driver's topics and types; (3) hardware-driver tests with
recorded or fake joint states and frames; (4) a supervised hardware smoke test with command
timeout, joint limits, emergency stop and stale-observation handling.

## Embodiment constraints

The SO-101 has five arm degrees of freedom and no shoulder roll, so its planner uses
approach-constrained waypoints instead of full six-degree-of-freedom poses; table waypoints
must stay near the tested surface and within reach. The simulation uses the new-calibration
SO-101 model; changes to its model files require rechecking pad geometry, joint limits,
contact behavior and reachable area.

## Known remaining gaps

- **Jog resolver construction.** `TwistToJointResolver` is built separately in
  `robots/so101/mujoco_env.py`, `robots/so101/shared.py` and `scripts/teleop_keyboard.py`,
  although the jog math is shared (`robots/so101/jog.py`).
- **Host mode branching.** `Host` still branches on single-robot versus shared world in about
  a dozen places, because only a single robot runs a policy, records or plays back. The
  branches go away when shared-world sessions gain tasks and policies, not by splitting `Host`
  (its lease, `web/lease.py`, and camera worker, `web/cameras.py`, are already separate).
- **`scripts/` composition-root adoption.** Scripts other than `run_sim.py` (`workspace_map`,
  `benchmark_ik`, `render_docs_media`, `teleop_keyboard`, `collect_demos`, `eval_policy`,
  `eval_randomization`, `plan_task`) still hand-assemble `EnvConfig` and environments instead
  of calling `create_runtime()` or `create_session()`. They are safety-gated either way, since
  the gate lives in the adapter.
- **Legacy configuration.** `--config`, `--world`, `configs/tasks/` and `configs/worlds/` are
  deprecated in favor of manifests and removed after a deprecation window;
  `scripts/run_ros2_sim.py --config` still reads the task file. Shared-world sessions run no
  tasks or policies yet.
- **Isaac Sim is single-cube and observation-only.** It supports the single-cube pick-and-place
  scene, headless or with `--serve` (a MuJoCo display mirror of the scene), a fixed target, a
  lighting scale and camera jitter, and observation-based policies. `--viewer`, shared worlds,
  the ROS2 bridge, web recording and playback, and `ArmKinematics`' full-scene collision queries
  are MuJoCo-only, and a manifest combining `simulator: isaac` with any of them is rejected at
  load time ([ADR 15](adr/simulators.md#adr-15-simulator-engine-selection)). `collect_demos.py`
  takes no `--sim` (it needs privileged MuJoCo state); `eval_policy.py --sim isaac` evaluates
  observation-based policies.
- **ROS2 Cartesian target service.** `SO101ROS2Node.handle_cartesian_target()` exists but is not
  bound to a ROS2 service or action endpoint.
