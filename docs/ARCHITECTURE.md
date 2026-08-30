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

## Module ownership

| Module | Owns | Must not own |
| --- | --- | --- |
| `physai.contracts` | Shared `Observation`, `Action`, and ROS2-shaped value types | Robot-specific ordering or task rules |
| `physai.robots` | Embodiment discovery, `RobotSpec`, factories, robot adapters, and robot-specific MuJoCo behavior | Task reward, planner decisions, or model SDKs |
| `physai.tasks` | Task state, reset rules, reward, metrics, and termination | Robot internals or action generation |
| `physai.sim` | Current MuJoCo scene/environment orchestration and compatibility facades | Generic robot capabilities or planner logic |
| `physai.planner` | Instruction and image grounding, `Plan`, and `SubGoal` production | Control-rate motor commands |
| `physai.policy` | Control-rate `Action` production, scripted policies, and model adapters | Task scoring or robot discovery |
| `physai.control` | Action resolution, capability checks, and rate limiting | High-level planning or task semantics |
| `physai.data` | Episode recording and dataset loading | Simulation decisions or model inference |
| `physai.bridge` | Future ROS2 topic and message mapping contracts | Phase 0 runtime ownership |
| `scripts/` | CLI argument parsing and runtime composition | IK, reward calculation, or SDK-specific implementation |
| `tests/` | Executable behavior and contract coverage | New runtime ownership |

The source tree follows this ownership map:

```text
src/physai/
├── contracts.py       shared message-shaped values
├── robots/            embodiment adapters and factories
├── tasks/             task rules and registries
├── sim/               MuJoCo scene/environment orchestration
├── planner/           language-to-plan implementations
├── policy/            control and model adapters
├── control/           action resolution and rate limiting
├── data/              episode recording and loading
└── bridge/            future ROS2 boundary
```

## Dependency direction

```text
scripts
      |
      v
composition: robot + task + planner/policy + backend
      |                 |                 |
      v                 v                 v
robots            tasks            approaches
      \                 |                 /
          +----------- contracts.py --------+
```

`scripts/` composes registered components. It should not know how a robot
computes IK, how a task calculates reward, or which SDK a model uses. The
shared contracts are the boundary between embodiments and approaches.

## Design patterns in use

- **Ports and Adapters:** `Observation`, `Action`, `RobotEnv`, `Planner`,
     `Policy`, and `Task` are ports. MuJoCo, ROS2, model SDKs, and concrete robot
     implementations are adapters around those ports.
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

These patterns are intentionally lightweight. A new abstraction is warranted
only when it removes coupling at a boundary or makes a component replaceable;
inheritance should not be added solely to make a class hierarchy.

Adding one component should not require changing an unrelated component:

- A robot owns its adapter, capabilities, factory, and registration.
- A task consumes backend state and owns task semantics.
- A planner returns a `Plan` and does not emit motor commands.
- A policy emits `Action` at the control rate and does not score the task.
- A model-specific observation or action space requires an explicit adapter.

## Stable contracts

`Observation` contains joint state, named image frames, optional end-effector
pose, step count, and simulation time. `Action` can carry either joint targets
or a Cartesian/base `Twist`, plus a normalized gripper command. The selected
robot capability decides which representation is valid; an action must not
silently carry both modes.

`RobotSpec` describes an embodiment without exposing simulator or hardware API:
its joint names, action modes, observation modalities, and named capabilities.
Workflow code should call `supports()` or `require()` rather than branch on a
robot name.

`Planner` maps an instruction and observation to `Plan`. A `Plan` contains
language-grounded `SubGoal` values and optional `PoseStamped` waypoints.
`Policy` maps an observation and optional goal to one `Action` per control tick.
`Task` owns evaluation, reward, and termination around the backend state.

## Runtime compositions

The currently supported compositions are:

```text
so101 + pick_place + scripted policy or Planner + PlanRunner + MuJoCo
turtlebot4 + generic smoke test + constant twist policy + MuJoCo
```

`so101` and `turtlebot4` are registered robots. `pick_place` is the registered
manipulation task and requires arm and gripper capabilities, so the policy,
demo, and planner workflows are currently SO-101-specific. TurtleBot4 has a
native MuJoCo model, differential-drive controls, wheel state, and base pose,
but no navigation task yet.

Future compositions such as
`turtlebot4 + navigation + Nav2 policy` or
`mobile_manipulator + mobile_pick_place + VLA` should add adapters, tasks, and
registrations. They should not route TurtleBot4 through `SO101PickPlaceEnv`,
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
The intended next integration target is ROS2 Jazzy on Ubuntu 24.04.

## Embodiment constraints

The SO-101 model has five arm degrees of freedom and no shoulder roll, so its
planner uses approach-constrained waypoints instead of full six-degree-of-
freedom poses. Table waypoints must remain near the tested surface and within
the reachable workspace.

The simulation uses the new-calibration SO-101 model. Changes to robot model
files require rechecking pad geometry, joint limits, contact behavior, and
reachable area.

## Research notes

- Rate-limit joint and gripper commands; direct step inputs can displace the
     cube.
- Position-only IK can produce a solution that is unreliable during descent.
- Flat jaw pads are more stable than spherical pads for the cube task.
- Post-grasp waypoints should use absolute heights instead of tracking the
     moving cube.
- Contact behavior and sim-to-real differences are likely to matter more than
     planner latency for this task.
