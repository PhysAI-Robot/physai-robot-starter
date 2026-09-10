# SO-101 Runbook

This is the end-to-end SO-101 workflow. Start with the MuJoCo robot and basic
task, then validate ROS2, and only then continue to imitation learning or
planner/VLM/VLA workflows.

## 1. Fetch and Open the Robot

```bash
uv run python scripts/fetch_assets.py --robot so101
```

Open the robot in MuJoCo:

```bash
uv run python scripts/run_sim.py \
  --robot so101 \
  --viewer \
  --seed 0
```

Open the checked-in pick-and-place scene:

```bash
uv run python scripts/run_sim.py \
  --config configs/tasks/so101/pick_place.yaml \
  --viewer \
  --seed 0
```

For headless execution:

```bash
uv run python scripts/run_sim.py \
  --config configs/tasks/so101/pick_place.yaml \
  --no-video \
  --seed 0 \
  --max-steps 500
```

Do not use `--viewer` in CI or on a headless machine. Omit `--no-video` when
you want the script to write video frames under `outputs/`.

## 2. Run the Basic Task

Run the scripted pick-and-place policy:

```bash
uv run python scripts/eval_policy.py \
  --policy scripted \
  --episodes 5 \
  --seed 0 \
  --max-steps 500
```

Run the documented deterministic reliability check:

```bash
uv run python scripts/eval_policy.py \
  --policy scripted \
  --episodes 20 \
  --seed 0 \
  --max-steps 600
```

The current baseline is `success 20/20`. This depends on the calibrated jaw
pads and deterministic scene; it is not a hardware or randomized result.

Use the same seed when comparing parameter changes. Save local results when
needed:

```bash
mkdir -p outputs/local
uv run python scripts/eval_policy.py \
  --policy scripted \
  --episodes 5 \
  --seed 0 \
  --max-steps 500 \
  --json-out outputs/local/so101_scripted_seed0.json
```

## 3. Validate SO-101 Contracts

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest \
  tests/test_sim.py tests/test_tasks.py tests/test_robot_registry.py -q
```

The transport-level ROS2 check does not require a ROS2 installation:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest \
  tests/test_robot_registry.py::test_ros2_mujoco_teleop_command_moves_so101 -q
```

## 4. Run the Real ROS2 Node

Real message and executor coverage requires ROS2 Jazzy:

```bash
source /opt/ros/jazzy/setup.bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest \
  tests/robots/so101/test_ros2_node.py -q
```

Run a bounded ROS2 MuJoCo smoke test:

```bash
source /opt/ros/jazzy/setup.bash
MUJOCO_GL=osmesa uv run python scripts/run_ros2_sim.py \
  --robot so101 \
  --config configs/tasks/so101/pick_place.yaml \
  --seed 0 \
  --max-ticks 500
```

The node accepts joint trajectory and gripper commands and publishes joint
states, camera frames, camera info, and the SO-101 TF tree. The fake transport
test verifies command translation; the real-node test verifies ROS2 message and
executor behavior. Neither proves hardware connectivity or production QoS.

Inspect all shared endpoints with:

```bash
uv run python scripts/show_ros2_contract.py
```

## 5. Continue to Imitation Learning

After the scripted task is reliable, inspect demonstration collection and ACT:

```bash
uv run python scripts/collect_demos.py --help
uv run python scripts/train_act.py --help
uv run python scripts/eval_policy.py --help
```

The current prototype supports ACT-shaped data and scripted, replay, and ACT
policy evaluation. It does not yet export the standard `LeRobotDataset` format
or provide the planned unified training entry point. This is the Phase 2
continuation of the SO-101 journey.

## 6. Continue to Planner, VLM, and VLA

Inspect the planner workflow after the low-level policy path is understood:

```bash
uv run python scripts/plan_task.py --help
```

The scripted planner is the deterministic starting point. SmolVLM and Claude
backends are optional. Planner output must remain a validated plan and must
pass robot capability and safety checks before producing commands.

## 7. Parameters and Open Work

Main files are `configs/tasks/so101/pick_place.yaml`,
`src/physai/robots/so101/env.py`, `src/physai/sim/scenes/common.py`, and
`src/physai/policy/scripted.py`.

Run the reproducible FK, Jacobian, and IK benchmark with:

```bash
uv run python scripts/benchmark_ik.py --targets 20 --seed 0
```

The benchmark reports success rate, position/orientation error, iterations,
FK/Jacobian/IK runtime, and Jacobian shape. Remaining SO-101 work is the
broader future-robot adapter support. Cartesian requests are available through
`SO101ROS2Node.handle_cartesian_target()` and return structured acceptance,
IK error, iteration, and failure-reason fields for wiring into a ROS2 service
or action interface. Controlled domain randomization is already available
through the shared simulation configuration.