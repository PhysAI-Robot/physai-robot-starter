# Architecture and Research Notes

## Runtime flow

```text
instruction + camera images
        |
        v
   VLM planner (SmolVLM or Claude)
        |
        v
   Plan: language sub-goals + PoseStamped waypoints
        |
        v
   low-level policy or PlanRunner
        |
        v
   absolute joint targets + gripper command
        |
        v
   MuJoCo environment
```

The current runnable baseline is `ScriptedPlanner` plus `PlanRunner`. The
SmolVLM and Claude planners cover the vision-and-language planning step. A
future VLA policy can consume the same camera, state, and task interfaces.

## Project structure

```text
physai-robot-starter/
├── src/physai/
│   ├── contracts.py          shared Observation and Action data types
│   ├── robots/               robot discovery and embodiment factories
│   │   ├── base.py           RobotSpec and RobotEnv contracts
│   │   ├── registry.py       create_robot("so101")
│   │   └── so101/            complete SO-101 implementation package
│   │       ├── factory.py    SO-101 environment factory
│   │       └── kinematics.py SO-101 FK, IK, and Jacobian
│   ├── tasks/                task rules, independent from robot and model
│   │   ├── base.py           Task contract
│   │   ├── registry.py       create_task("pick_place")
│   │   └── pick_place.py     cube pick-and-place rules
│   ├── sim/                  MuJoCo backend for the current task
│   │   ├── env.py            SO101PickPlaceEnv compatibility facade
│   │   └── scene.py           SO-101 pick-place world construction
│   ├── planner/              language and vision planning approaches
│   │   ├── base.py           Planner, Plan, and SubGoal contracts
│   │   ├── registry.py       planner backend discovery
│   │   ├── smolvlm.py        local SmolVLM backend
│   │   └── claude_vlm.py     Claude backend
│   ├── policy/               control approaches and model adapters
│   │   ├── base.py           Policy contract
│   │   ├── scripted.py       SO-101 privileged expert
│   │   ├── plan_runner.py    waypoint-to-action baseline
│   │   └── vla_adapter.py    LeRobot/VLA adapter
│   ├── control/              action resolvers and rate limiting
│   ├── data/                 episode recording and loading
│   └── bridge/               future ROS2 topic contracts
├── scripts/                  thin command-line entry points
├── configs/                  task and simulation reference values
├── tests/                    unit, integration, and simulation tests
└── docs/                     architecture and research notes
```

### Dependency direction

```text
scripts
     |
     v
runtime composition: robot + task + planner/policy + backend
     |                  |          |
     v                  v          v
robots            tasks     approaches
     \                  |          /
      +---------- contracts.py ---+
```

The important rule is that `scripts/` should compose components; it should not
know how a robot computes IK, how a task calculates reward, or which SDK a
model uses. Robot-specific implementation belongs in `robots/` or its backend,
task-specific rules belong in `tasks/`, and model decisions belong in
`planner/` or `policy/`.

### Current implementation status

The boundaries are in place, but the first implementation is still intentionally
small. `so101` is the only registered robot, `pick_place` is the only registered
task, and the MuJoCo backend currently constructs the SO-101 pick-and-place
world. The CLI exposes `--robot` and `--planner`; task selection is currently an
API/configuration concern through `EnvConfig.task` and `create_task()`.

This means a TurtleBot should get its own robot package, motion capabilities,
backend, and navigation task. It should not be forced through
`SO101PickPlaceEnv`, `ArmKinematics`, or `PickPlaceTask`.

### Current SO-101 composition

```text
so101 robot factory
                    + pick_place task
                    + scripted policy or SmolVLM planner + PlanRunner
                    + MuJoCo backend
```

Future compositions can be `turtlebot + navigation task + Nav2 policy` or
`mobile_manipulator + mobile_pick_place task + fine-tuned VLA`. Those additions
should implement new adapters and registrations rather than modify shared
planner, policy, or contract interfaces.

## Kinematics and motion capabilities

`src/physai/robots/so101/kinematics.py` is **SO-101-specific**. It should not
be treated as the generic kinematics module for every robot. There is no
generic kinematics implementation in `sim/`.

```text
generic capability contract
     |
     +-- SO101Kinematics       FK, IK, Jacobian, gripper conversion
     +-- TurtleBotMotion       base velocity, odometry, no arm IK
     +-- MobileManipulatorIK   base motion + arm IK + gripper
```

The generic layer should describe capabilities, not force one action model on
every embodiment. A mobile base may provide `base_velocity` and `odometry`, an
arm may provide `joint_position` and `end_effector_pose`, and a mobile
manipulator may provide both. Robot adapters should expose only the capabilities
they support; tasks and policies can then declare which capability they need.

## Extension boundaries

The intended dependency direction is:

```text
robot adapter -> generic Observation/Action contracts <- planners and policies
                              ^
                     task/environment orchestration
```

- `physai.robots` owns embodiment discovery and factories.
- `physai.sim` currently contains the SO-101 MuJoCo implementation; it is not
     yet a generic simulation backend.
- `physai.tasks` owns task rules such as pick-and-place metrics, reward, and
  termination; it consumes backend state instead of robot internals.
- `physai.planner` owns language-to-plan backends and the `Plan` contract.
- `physai.policy` owns control-rate action backends and the `Policy` contract.
- `physai.contracts` contains the stable message-shaped data shared by all
     approaches.

Adding a robot should not require changing a task, planner, or policy. Adding a
task should not require changing a robot. Adding a planner should not require
changing a robot. When a model requires a different observation or action
space, add an explicit adapter or capability to the robot spec rather than
silently reshaping arrays in shared code.

## Model roles

SmolVLM is the local high-level VLM. It reads the front and wrist camera frames
and returns a structured pick-and-place plan. Claude is an optional cloud
backend for the same planner interface.

SmolVLA is a different model class: it is a low-level vision-language-action
policy that predicts joint actions at the control rate. It should not be
confused with SmolVLM. The adapter in
`src/physai/policy/vla_adapter.py` defines the policy boundary.

## Data contract

Recorded episodes use LeRobot-shaped keys:

```text
observation.images.front  (T, H, W, 3) uint8
observation.images.wrist  (T, H, W, 3) uint8
observation.state         (T, 6) float32, radians
action                    (T, 6) float32, absolute joint targets
```

The six values are ordered as:
`shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`.

## Robot constraints

The SO-101 model has five arm degrees of freedom and no shoulder roll. The
planner therefore uses approach-constrained waypoints rather than demanding a
full six-degree-of-freedom pose. Keep table waypoints close to the surface and
within the tested workspace range.

The simulation uses the new-calibration SO-101 model. Changing robot model
files requires rechecking the pad geometry, joint limits, and reachable area.

## ROS2 boundary

The message-shaped contracts are defined in
`src/physai/contracts.py`; planned ROS2 topics are documented in
`src/physai/bridge/ros2_contract.py`. Phase 0 runs in one process and does not
require ROS2. The intended next integration target is ROS2 Jazzy on Ubuntu
24.04.

## Experiment notes

- Rate-limit joint and gripper commands; direct step inputs can knock the cube
  away.
- Position-only IK can report a solution that the physical arm cannot reach
  reliably during descent.
- Flat jaw pads are more stable than spherical pads for the cube task.
- Post-grasp waypoints should use absolute heights instead of tracking the
  moving cube.
- Contact behavior and sim-to-real differences are likely to matter more than
  planner latency for this task.
