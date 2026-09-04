# Phase 1 Runbook

This guide explains how to run the Phase 1 baseline through `uv`, reproduce the test results, evaluate the scripted SO-101 workflow, and edit the most relevant parameters.

This document reflects the current state of branch `feat/phase1-foundation-ci`:

- Test suite: 71 passed, 3 skipped on a host without `control_msgs` or PyTorch.
- SO-101 ROS2 teleoperation through fake transport and a real `rclpy` node: covered by acceptance tests when Jazzy message packages are installed.
- TurtleBot4 Nav2 and the domain-randomization engine: not complete.
- Scripted SO-101 pick-and-place reaches `success 20/20` over seeds 0-19 (`--episodes 20 --seed 0 --max-steps 600`) in the checked-in deterministic scene. The result depends on the calibrated pad geometry: the added pads are placed on the actual jaw contact surfaces and replace the original jaw collision meshes, so the mesh and pad do not compete for contact resolution. This is a deterministic baseline result, not a claim about domain-randomized or hardware runs.

## 1. Prerequisites

Supported project baseline:

- Ubuntu 24.04
- Python 3.12
- MuJoCo and the project's base dependencies
- ROS2 Jazzy only for real ROS2 and hardware workflows; direct simulation and tests do not require it

Run all commands below from the repository root:

```bash
cd /path/to/physai-robot-starter
```

## 2. Set Up the uv Environment

Synchronize the project and development dependencies:

```bash
uv sync --locked --extra dev
```

Run commands through the project environment:

```bash
uv run python --version
uv run python -m pytest --version
```

The project requires Python `3.12.x`. `uv` creates and manages `.venv`
automatically; it does not need to be activated manually.

## 3. Baseline Tests

Run the full regression suite before and after editing code:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/ -q
```

Expected result on this host:

```text
71 passed, 3 skipped
```

With `ros-jazzy-control-msgs` installed, the ROS2 acceptance test runs and the
result becomes `72 passed, 2 skipped`.

If the test count changes, focus on the exit code and failure details, not only the test count.

The most relevant tests for current Phase 1 work are:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest \
  tests/test_sim.py tests/test_tasks.py \
  tests/test_robot_registry.py::test_ros2_mujoco_teleop_command_moves_so101 -q
```

## 4. Fetch SO-101 Assets

Fetch the robot asset if it is not already available:

```bash
uv run python scripts/fetch_assets.py --robot so101
```

Assets are placed under `assets/so101/`. Do not use this folder for model snapshots or generated experiment output that should be committed.

## 5. Run SO-101 Direct MuJoCo

The `--config` argument is optional. Without it, `run_sim.py` uses its built-in
SO-101 defaults. Add `--no-video` when you only want the simulation result and
do not want frames written under `outputs/`.

Run quickly without video using the built-in defaults, which is useful for debugging:

```bash
uv run python scripts/run_sim.py \
  --robot so101 \
  --no-video \
  --seed 0 \
  --max-steps 500
```

Run with video output under `outputs/` using the built-in defaults:

```bash
uv run python scripts/run_sim.py \
  --robot so101 \
  --seed 0 \
  --max-steps 500
```

Run without video using the checked-in task configuration:

```bash
uv run python scripts/run_sim.py \
  --config configs/task_pick_place.yaml \
  --no-video \
  --seed 0 \
  --max-steps 500
```

Open the interactive viewer using the built-in defaults. The viewer does not
write a video file:

```bash
uv run python scripts/run_sim.py \
  --robot so101 \
  --viewer \
  --seed 0
```

To open the viewer with the checked-in task configuration, include the config
explicitly:

```bash
uv run python scripts/run_sim.py \
  --config configs/task_pick_place.yaml \
  --viewer \
  --seed 0
```

Do not use `--viewer` in headless or CI workflows.

## 6. Evaluate Scripted Pick-and-Place

Repeat the first five seeds:

```bash
uv run python scripts/eval_policy.py \
  --policy scripted \
  --episodes 5 \
  --seed 0 \
  --max-steps 500
```

Run a longer evaluation:

```bash
uv run python scripts/eval_policy.py \
  --policy scripted \
  --episodes 20 \
  --seed 0 \
  --max-steps 600
```

Interpret the result as follows:

- `success=True` means the task reached the target and held it for `success_hold_steps`.
- `steps=500` or `steps=600` means the episode reached its time limit; inspect the policy phase and cube position.
- `success 20/20` is the current verified deterministic baseline over seeds 0-19. Any lower rate is not a pytest failure; it is a task-reliability failure that must be investigated separately.
- Use the same seed when comparing parameter changes.

To save evaluation results as local JSON:

```bash
mkdir -p outputs/local
python scripts/eval_policy.py \
  --policy scripted \
  --episodes 5 \
  --seed 0 \
  --max-steps 500 \
  --json-out outputs/local/scripted_seed0.json
```

The `outputs/` directory and local evaluation results do not need to be committed.

## 7. Reproduce ROS2-Shaped SO-101 Teleoperation

The currently tested path uses `RecordingTransport`, so it does not require a ROS2 installation. The test sends both commands through the same endpoints used by the ROS2 contract:

- `/arm_controller/joint_trajectory`
- `/gripper_controller/gripper_cmd`

Run the acceptance test:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest \
  tests/test_robot_registry.py::test_ros2_mujoco_teleop_command_moves_so101 -q
```

The test verifies that:

1. SO-101 is created with the `ros2_mujoco` adapter.
2. The trajectory and gripper callbacks accept commands.
3. The commands run for multiple control ticks.
4. The arm joint positions move toward the requested target.
5. The gripper position changes as well.

Print the ROS2-shaped endpoint list with:

```bash
uv run python scripts/show_ros2_contract.py
```

The fake-transport path does not prove ROS2 QoS, serialization, executor
behavior, or hardware connectivity. Use the real node and acceptance test
below for the ROS2 message and executor path.

The repository also provides a real ROS2 SO-101 MuJoCo node. It requires a
sourced ROS2 Jazzy environment and the `control_msgs` package:

```bash
source /opt/ros/jazzy/setup.bash
ros2 run --help >/dev/null
uv run python scripts/run_ros2_sim.py \
  --config configs/task_pick_place.yaml \
  --seed 0
```

The command runs continuously until ROS2 is shut down. For a real-time
headless control loop, disable camera rendering and optionally bound the run:

```bash
MUJOCO_GL=osmesa uv run python scripts/run_ros2_sim.py \
  --config configs/task_pick_place.yaml \
  --seed 0 \
  --no-camera \
  --max-ticks 500
```

Camera rendering is useful for image-topic consumers but is substantially more
expensive than the MuJoCo control tick on software-rendered hosts.

The node subscribes to the trajectory and gripper topics, publishes joint
states and camera images, publishes matching `CameraInfo`, and broadcasts the
MuJoCo body and camera frames on `/tf`. Run the real-message acceptance test
in the same ROS2 environment:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest \
  tests/test_ros2_node.py -q
```

The Docker image installs `ros-jazzy-control-msgs`. On a host installation,
install that package with the system ROS2 package manager before running the
node or acceptance test.

## 8. Parameters to Edit

### Global simulation parameters

File: `configs/sim_config.yaml`

```yaml
seed: 0
domain_randomization:
  enabled: false
```

- `seed`: default simulation seed.
- `domain_randomization.enabled`: must remain `false` because the randomization engine is not implemented yet.

### Task and scene parameters

File: `configs/task_pick_place.yaml`

Parameters suitable for experiments:

- `scene.table_pos`: table position.
- `scene.table_size`: table dimensions.
- `scene.target_pos`: target position.
- `scene.target_radius`: visual target radius.
- `scene.cube_pos`: initial cube position when randomization is disabled.
- `scene.cube_mass`: cube mass.
- `env.control_hz`: control rate. Default is `25` Hz.
- `env.max_steps`: maximum episode length.
- `env.randomize_cube`: enable or disable cube-position randomization.
- `env.cube_x_range`, `env.cube_y_range`: cube-position ranges.
- `env.randomize_target`: enable or disable target randomization.
- `env.success_xy_tol`: task horizontal-distance tolerance.
- `env.success_hold_steps`: number of ticks for which the target must remain satisfied before success.

After changing the scene or cube position, validate with a fixed seed:

```bash
uv run python scripts/run_sim.py \
  --config configs/task_pick_place.yaml \
  --no-video \
  --seed 0 \
  --max-steps 500
```

File: `src/physai/robots/so101/env.py`, class `EnvConfig`.

- `gripper_force_limit`: caps the gripper actuator's `forcerange` (N*m). The
  XML default (+/-3.35) is a per-servo torque rating, not a grip-force budget,
  and lets a fixed-position squeeze overload the pad-cube contact; the default
  here (`0.3`) emulates a current-limited real servo.

File: `src/physai/sim/scenes/common.py`, class `ManipulationSceneConfig`.

- `static_pad_pos` and `moving_pad_pos`: local pad positions calibrated to the
  actual jaw contact surfaces. The checked-in defaults are part of the
  deterministic pick-and-place baseline.
- `replace_jaw_collision`: when `true`, disables the original group-3 jaw mesh
  collisions so the calibrated pad geoms are the only jaw contacts. Set it to
  `false` only when supplying a replacement scene with its own collision
  calibration.

### Scripted policy parameters

File: `src/physai/policy/scripted.py`, class `ExpertConfig`.

Parameters controlling the motion:

- `hover_height`: approach height above the cube.
- `grasp_height`: pinch-center height during grasp.
- `lift_height`: height used while carrying the cube.
- `place_height`: offset used while lowering the cube onto the target.
- `gripper_open`: initial aperture.
- `gripper_touch`: aperture used to capture the cube.
- `gripper_grip`: squeeze aperture after capture.
- `pos_tol`: waypoint position tolerance.
- `settle_steps`: number of ticks to wait for the gripper to settle.
- `max_joint_rate`: joint rate during approach.
- `approach_rate`: joint rate during descent and precision phases.
- `gripper_rate`: aperture change rate per second.

When tuning the grasp, change one parameter per experiment and use the five-seed evaluation command. Do not immediately increase `pos_tol` to hide a contact failure. Check whether the cube is actually lifted and remains held during transfer.

### Adapter and ROS2 parameters

File: `src/physai/bridge/ros2_contract.py`

- Topic names, message types, rates, owners, and unit notes are defined in `ROBOT_ENDPOINTS`.
- Frame names are defined in `TF_FRAMES`.

File: `src/physai/bridge/adapters.py`

- `_ROS2RobotAdapter` receives trajectory and gripper commands.
- `publish_observation()` publishes joint state and camera frames.
- Physics and task reward do not belong in the bridge module.

File: `src/physai/bridge/mujoco_ros_bridge.py`

- `MuJoCoROSBridge.tick()` runs one control period.
- `MuJoCoROSBridge.run()` runs a loop using `control_hz` from `RobotSpec` or an override.
- `RclpyTransport` is only a transport adapter; the calling application remains responsible for importing and configuring `rclpy`.

## 9. Edit and Validation Workflow

Use this sequence when changing parameters or code:

1. Record a baseline test and evaluation with the same seed.
2. Change one owning module or one parameter group.
3. Run the focused test closest to the change.
4. Run scripted evaluation if the change touches the scene, kinematics, gripper, or policy.
5. Run the full suite.
6. Inspect the change:

```bash
git status --short
git diff --check
git diff --stat
```

7. Confirm that `.venv`, model snapshots, demonstrations, videos, and credentials are not included in the diff.

Example validation after changing SO-101 policy code:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest \
  tests/test_sim.py tests/test_tasks.py \
  tests/test_robot_registry.py::test_ros2_mujoco_teleop_command_moves_so101 -q

uv run python scripts/eval_policy.py \
  --policy scripted \
  --episodes 5 \
  --seed 0 \
  --max-steps 500

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/ -q
```

## 10. Current Phase 1 Boundaries

The following items remain separate work and must not be considered complete only because the commands above pass:

- The SO-101 TF tree is not published.
- Real `rclpy` node integration is not tested.
- TurtleBot4 does not yet have a Nav2 goal workflow.
- An IK benchmark with success rate, error, iterations, and runtime metrics is not available.
- The domain-randomization engine is not available.
- Scripted pick-and-place reaches the reliability target on the first five
  seeds and on the documented 20-seed check with the calibrated pad setup.
