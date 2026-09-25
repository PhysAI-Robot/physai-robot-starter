# SO-101 Runbook

This is the end-to-end SO-101 workflow. Start with the MuJoCo robot and basic
task, then validate ROS2, and only then continue to imitation learning or
planner/VLM/VLA workflows.

## 1. Fetch and Open the Robot

```bash
uv run python scripts/fetch_assets.py --robot so101
```

The fetch includes the upstream `so101_new_calib_camera.xml` variant and its
wrist camera mount meshes. When that file is present, the simulator selects it
automatically; otherwise it falls back to the base SO-101 model.

Open the robot interactively in the browser (see the shared
[Web Viewer Runbook](WEB_VIEWER_RUNBOOK.md) for the full workflow, keyboard
mapping, and camera-panel configuration):

```bash
uv run python scripts/run_sim.py --robot so101 --serve --seed 0
```

Open the checked-in pick-and-place scene the same way:

```bash
uv run python scripts/run_sim.py --config configs/tasks/so101/pick_place.yaml --serve --seed 0
```

For headless execution:

```bash
uv run python scripts/run_sim.py --config configs/tasks/so101/pick_place.yaml --seed 0 --max-steps 500
```

Video recording is opt-in. Add `--video` when you want frames written under
`outputs/`; use `--serve` (or the native viewer below) only for interactive
local runs.

MuJoCo's own desktop viewer is also available as a fallback when a browser
isn't convenient — see [ADR 4](adr/0004-tk-viewer-frozen.md). It
opens a native window with MuJoCo's built-in scene navigation (drag to orbit,
scroll to zoom); combine it with `--serve` and use the browser viewer's
camera grid for per-camera views or debug overlays:

```bash
uv run python scripts/run_sim.py --config configs/tasks/so101/pick_place.yaml --viewer --seed 0
```

This mode requires a desktop display.

## 2. Run the Basic Task

The scripted pick-and-place policy is a privileged-ground-truth expert — see
[research/scripted_experts/README.md](../research/scripted_experts/README.md)
for the full run/evaluate workflow, reliability numbers, and interactive
inspection command.

## 3. Validate SO-101 Contracts

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/acceptance/so101/test_kinematics.py tests/acceptance/so101/test_scene.py tests/unit/test_robot_registry.py -q
```

The transport-level ROS2 check does not require a ROS2 installation:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/integration/test_ros2_adapters.py::test_ros2_mujoco_teleop_command_moves_so101 -q
```

## 4. Run the Real ROS2 Node

Real message and executor coverage requires ROS2 Jazzy:

```bash
source /opt/ros/jazzy/setup.bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/robots/so101/test_so101_ros2_node.py -q
```

Run a bounded ROS2 MuJoCo smoke test:

```bash
source /opt/ros/jazzy/setup.bash
MUJOCO_GL=egl uv run python scripts/run_ros2_sim.py --robot so101 --config configs/tasks/so101/pick_place.yaml --seed 0 --max-ticks 500
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

After the scripted task is reliable, collect demonstrations and fine-tune
ACT — see
[research/imitation_learning/README.md](../research/imitation_learning/README.md)
for the full collect -> train -> evaluate workflow. This is the Phase 2
continuation of the SO-101 journey.

## 6. Continue to the planner contract

Inspect the planner workflow after the low-level policy path is understood:

```bash
uv run python scripts/plan_task.py --help
```

The scripted planner is the only backend that ships. A model-backed planner
implements the same `Planner` contract. Planner output must remain a validated
plan and must pass robot capability and safety checks before producing
commands.

## 7. Phase 2A Visual Servoing Baseline

The deterministic `visual_servo` policy is a classical, model-free baseline —
see
[research/classical_control/README.md](../research/classical_control/README.md)
for the full workflow, calibration notes, and CI evaluation reference.

## 8. Parameters and Open Work

Main files are `configs/tasks/so101/pick_place.yaml`,
`src/physai/robots/so101/env.py`, `src/physai/sim/scenes/common.py`, and
`research/scripted_experts/so101_pick_place_expert.py`.

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