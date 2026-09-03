# Architecture

This is the internal design reference for the runtime. It defines the
composition model, module ownership, stable contracts, and extension rules.
User setup and runnable commands live in [README.md](../README.md). Agent
workflow rules live in [AGENTS.md](../AGENTS.md).

## Design shape

The stack separates embodiment, task, and decision-making so each can evolve
independently. A runtime composition selects one robot, one task, and one
planner or policy, then connects them through message-shaped contracts.

```text
instruction + camera images
                    |
                    v
     Planner (SmolVLM, Claude, or scripted)
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

The current runnable baseline is the scripted planner/policy path with
`PlanRunner`; model-backed planners and VLA policies plug into the same
boundaries.

## Execution modes and deployment parity

MuJoCo and ROS2 are different layers, not competing robot backends. MuJoCo
provides simulation and physics; ROS2 provides process and device transport.
The project should support three execution paths with different purposes:

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

The proposed adapter boundary is:

```text
RobotPort
     +-- DirectMuJoCoAdapter
     +-- ROS2MuJoCoAdapter
     +-- ROS2HardwareAdapter
```

The adapters are generic at the port level and receive an embodiment-specific
mapping and configuration. `DirectMuJoCoAdapter` and `ROS2MuJoCoAdapter` use
MuJoCo as the simulation engine, while `ROS2HardwareAdapter` connects the
selected embodiment to its real device driver. The embodiment mapping changes
inside each adapter without changing `RobotPort`.

`DirectMuJoCoAdapter` may retain a synchronous `reset`/`step` API for fast local
execution. The two ROS2 adapters may use asynchronous callbacks and queues
internally, but they must translate to the same `Observation` and `Action`
contracts at the application boundary. `ROS2MuJoCoAdapter` is therefore an
integration adapter around MuJoCo, not a replacement for MuJoCo.

### Mode trade-offs

| Mode | Primary value | Residual risk |
| --- | --- | --- |
| Direct MuJoCo | Fast, deterministic iteration and CI | Does not exercise ROS2 timing, QoS, TF, or serialization |
| MuJoCo through ROS2 | Tests the production message and process boundary before hardware | Adds setup, scheduling, latency, and transport failure modes |
| Hardware through ROS2 | Tests real sensors, actuators, calibration, and safety behavior | Slow, nondeterministic, hardware-dependent, and higher risk |

The validation policy should keep direct MuJoCo in normal unit and regression
tests, while requiring the ROS2 MuJoCo path before hardware experiments. This
keeps development fast without allowing the direct path to hide integration
errors.

## Module ownership

| Module | Owns | Must not own |
| --- | --- | --- |
| `physai.contracts` | Shared `Observation`, `Action`, and ROS2-shaped value types | Robot-specific ordering or task rules |
| `physai.config` | Typed YAML configuration for shared simulation and task runtime settings | Simulation behavior, robot construction, or task evaluation |
| `physai.robots` | Embodiment discovery, `RobotSpec`, robot ports, factories, environments, and robot-specific adapters | Task reward, planner decisions, or model SDKs |
| `physai.tasks` | Task state, reset rules, reward, metrics, and termination | Robot internals or action generation |
| `physai.sim` | MuJoCo simulation core, generic scene primitives, task-specific scene builders, rendering, and simulation time | Robot-specific environment logic, ROS2 transport, QoS, or callbacks |
| `physai.planner` | Instruction and image grounding, `Plan`, and `SubGoal` production | Control-rate motor commands |
| `physai.policy` | Control-rate `Action` production, scripted policies, and model adapters | Task scoring or robot discovery |
| `physai.control` | Action resolution, capability checks, and rate limiting | High-level planning or task semantics |
| `physai.data` | Episode recording and dataset loading | Simulation decisions or model inference |
| `physai.bridge` | ROS2 transport, topic/message mapping, timing, and ROS2-backed adapters | Physics implementation, task semantics, or model inference |
| `physai.runtime` | Runtime composition, compatibility checks, and safety orchestration | Robot-specific physics, task reward, or model inference |
| `scripts/` | CLI argument parsing and runtime composition | IK, reward calculation, or SDK-specific implementation |
| `tests/` | Executable behavior and contract coverage | New runtime ownership |

The source tree follows this ownership map:

```text
src/physai/
├── contracts.py       shared message-shaped values
├── config.py          typed YAML runtime configuration
├── robots/            embodiment ports, adapters, and factories
│   ├── so101/         SO-101 environment and kinematics implementation
│   └── turtlebot/     TurtleBot4 environment and differential-drive implementation
├── tasks/             task rules and registries
├── sim/               MuJoCo simulation core and scene/environment orchestration
│   └── scenes/        shared world builder and task-specific scene variants
├── planner/           language-to-plan implementations
├── policy/            control and model adapters
├── control/           action resolution and rate limiting
├── data/              episode recording and loading
├── bridge/            ROS2 transport, message mapping, adapters, and tick loop
└── runtime/           robot-task-policy composition and safety orchestration
```

### Adapter dependency rule

```text
DirectMuJoCoAdapter  -> MuJoCoSimulationCore
ROS2MuJoCoAdapter    -> ROS2 transport + MuJoCoSimulationCore
ROS2HardwareAdapter  -> ROS2 transport + hardware port
```

`physai.sim` owns `MuJoCoSimulationCore` and must not import ROS2. Robot
environments and embodiment mappings live under `physai.robots`; the direct
adapter may live with that embodiment integration, while
`ROS2MuJoCoAdapter` and `ROS2HardwareAdapter` belong to the ROS2 boundary under
`physai.bridge`. Robot-specific joint mapping, capabilities, calibration, and
kinematics are supplied by `physai.robots` to all three adapters.

This split keeps ROS2 optional for fast MuJoCo tests and prevents transport
concerns from leaking into the simulator. `scripts/` selects the adapter in
the composition root; policy and task code depend only on shared contracts and
capabilities, not on a robot name or transport implementation.

`MuJoCoROSBridge` owns the synchronous control tick around
`ROS2MuJoCoAdapter`. `RclpyTransport` adapts an existing `rclpy` node to the
transport port without importing ROS2 from the core package. The current bridge
publishes joint states and available camera images and accepts joint trajectory
and gripper commands; TF, CameraInfo, and mobile-base endpoints remain later
Phase 1 work.

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

Each scene builder owns model geometry and initial object layout. Generic world
settings are separated from manipulation attachment settings; the built-in
manipulation config supplies SO-101 defaults, including calibrated pad positions
and replacement of the original jaw collision meshes, while another arm should
provide its own attachment config or builder. Robot model paths, end-effector
anchors, and pad attachment bodies are configuration, not hardcoded task
ownership. It must not own task reward, policy decisions, or ROS2 transport. The legacy `sim/scene.py`
facade may translate the old `SceneConfig(num_cubes=...)` API, but new code
should select `PickPlaceMinimalSceneConfig` or `SortingMinimalSceneConfig`
directly. This keeps task-specific object branches out of the shared scene
builder and makes a future task scene additive.

## Design patterns in use

- **Ports and Adapters:** `Observation`, `Action`, `RobotPort`, `Planner`,
     `Policy`, and `Task` are ports. Direct MuJoCo, ROS2 transport, model SDKs,
     and concrete robot implementations are adapters around those ports.
- **Strategy:** planners, policies, and tasks are interchangeable strategies.
     The caller depends on the abstract contract and selects the behavior at
     runtime.
- **Registry plus Factory:** robot, task, planner, and policy registries map
     stable names to factories. Lazy builtin loading keeps optional dependencies
     out of model-free workflows and lets extensions register without editing a
     caller.
- **Composition Root:** CLI modules under `scripts/` assemble the selected
     adapters and dependencies. Domain modules do not parse CLI arguments or
     discover unrelated implementations.
- **Safety Gate:** `SafetyController` validates action mode, joint order,
     finite values, timestamps, joint limits, and configured per-joint step
     limits immediately before a robot port receives a command.

These patterns are intentionally lightweight. A new abstraction is warranted
only when it removes coupling at a boundary or makes a component replaceable;
inheritance should not be added solely to make a class hierarchy.

Adding one component should not require changing an unrelated component:

- A robot owns its adapter, capabilities, factory, and registration.
- A task consumes backend state and owns task semantics.
- A planner returns a `Plan` and does not emit motor commands.
- A policy emits `Action` at the control rate and does not score the task.
- A model-specific observation or action space requires an explicit adapter.
- A simulator adapter and a hardware adapter implement the same robot port;
     neither is selected by branching inside a policy.
- ROS2 transport is optional for the direct fast path but mandatory for the
     ROS2 simulation and hardware deployment paths.

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

`Action` may carry explicit joint names and a timestamp. Legacy callers may
omit those fields, but adapters and safety-critical paths should populate them
so joint ordering and command freshness are checked at the boundary.

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
Phase 0 workflows.

`Planner` maps an instruction and observation to `Plan`. A `Plan` contains
language-grounded `SubGoal` values and optional `PoseStamped` waypoints.
`Policy` maps an observation and optional goal to one `Action` per control tick.
`Task` owns evaluation, reward, and termination around the backend state.

`physai.runtime.create_runtime` is the P0 composition entry point. It validates
the task's declared capabilities against the selected `RobotSpec`, wraps the
robot port with `TaskRuntime` when a task is selected, creates an optional
registered policy, and routes actions through `SafetyController` before
forwarding them to the robot port.

Scene configs are selected through `physai.sim.scenes.create_scene` and the
scene registry. Each registered scene declares supported robot kinds and task
names. `create_runtime(scene_name=...)` resolves that config before building
the robot and rejects task-scene mismatches during composition. The legacy
`SceneConfig(num_cubes=...)` facade remains available for compatibility.

## Runtime compositions

### Implemented compositions

The currently implemented compositions are concrete and should remain listed
here as an inventory of repository support:

```text
so101 + pick_place + scripted policy or Planner + PlanRunner + MuJoCo
turtlebot4 + generic smoke test + constant twist policy + MuJoCo
```

`so101` and `turtlebot4` are examples of registered embodiments, not names
that belong in the generic adapter or policy contracts. `pick_place` is the
registered manipulation task and requires arm and gripper capabilities, so the
policy, demo, and planner workflows are currently SO-101-specific. TurtleBot4
has a native MuJoCo model, differential-drive controls, wheel state, and base
pose, but no navigation task yet.

The current pick-and-place and sorting implementations are intentionally
minimal baselines for smoke tests and early experiments. They live in
`physai.tasks.pick_place_minimal` and `physai.tasks.sorting_minimal`; the
registry keys `pick_place` and `sorting` remain stable so configuration does
not encode an implementation filename.

The near-term instantiation is `so101 + pick_place`, but adding another robot
must not require changing this composition model. Every robot/task combination
should use the three execution paths defined in [Execution modes and
deployment parity](#execution-modes-and-deployment-parity); only the two ROS2
paths are deployment paths.

Future compositions such as
`turtlebot4 + navigation + Nav2 policy` or
`mobile_manipulator + mobile_pick_place + VLA` should add adapters, tasks, and
registrations. They should not route TurtleBot4 through an SO-101 environment,
SO-101 kinematics, or `PickPlaceTask`.

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

## Model roles

SmolVLM is a local high-level VLM that reads simulated camera frames and
returns a structured plan. Claude is an optional cloud backend for the same
planner contract.

SmolVLA and TurboVLA are low-level vision-language-action policy checkpoints.
They predict control-rate actions and belong behind the policy boundary in
`src/physai/policy/vla_adapter.py`. They must not be described as planners or
be coupled directly to a robot environment.

Model snapshots belong in the ignored local `models/` directory and are loaded
through an explicit local path. Model storage and path resolution are owned by
`src/physai/model_store.py`.

## Demonstration data

SO-101 demonstrations use LeRobot-shaped arrays:

```text
observation.images.front  (T, H, W, 3) uint8
observation.images.wrist  (T, H, W, 3) uint8
observation.state         (T, 6) float32, radians
action                    (T, 6) float32, absolute joint targets
```

The six values are ordered as:
`shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`.
Dataset recording and loading belong to `physai.data`; task semantics do not.

## ROS2 boundary

Phase 0 runs in one process without ROS2. The message-shaped values in
`src/physai/contracts.py` intentionally match the planned ROS2 types, while
the topic mapping is documented in `src/physai/bridge/ros2_contract.py`.
The first synchronous bridge core is available in
`src/physai/bridge/mujoco_ros_bridge.py`; the integration target is ROS2 Jazzy
on Ubuntu 24.04.

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
default `ContractMessageCodec` is used by Phase 0 tests, while
`ROS2MessageCodec` can construct real ROS2 message instances without adding
`rclpy` as a core dependency.

Both adapters must make unit conversion, joint ordering, timestamps, frame
names, command freshness, and command rate explicit. Joint order and value
shape are validated at decode time against `RobotSpec`. The current adapter
publishes only the observation fields represented by the Phase 0 contract;
CameraInfo, TF, and mobile-base endpoint publication require the remaining
Phase 1 integration work. The ROS2 contract file is the source of truth for
those external interfaces; it is not itself an adapter.

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
