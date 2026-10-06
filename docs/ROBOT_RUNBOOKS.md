# Robot runbooks

End-to-end workflows for each supported robot. Start with the MuJoCo robot and basic
task, validate ROS2, then continue to research. The browser viewer's controls,
recording and troubleshooting are in the [Web Viewer runbook](WEB_VIEWER_RUNBOOK.md);
this file only says how to start it.

## SO-101

### Fetch and open

```bash
uv run python scripts/fetch_assets.py --robot so101
uv run python scripts/run_sim.py --serve --seed 0
uv run python scripts/run_sim.py --manifest configs/manifests/so101_single_cube_fixed_place.yaml --serve --seed 0
uv run python scripts/run_sim.py --manifest configs/manifests/so101_single_cube_fixed_place.yaml --seed 0 --max-steps 500
```

The fetch includes the upstream `so101_new_calib_camera.xml` variant and its wrist
camera mount meshes; the simulator selects that file when present and falls back to the
base model otherwise. The first command opens an idle robot in the browser, the second
the checked-in pick-and-place scene, the third runs it headless. Video is opt-in:
`--video` writes frames under `outputs/`. MuJoCo's own desktop viewer
(`--viewer`, needs a display; combine with `--serve` for the camera grid) is a fallback
([ADR 4](adr/web-host.md#adr-4---viewer-is-mujocos-own-viewer-frozen-in-scope)).

The scripted pick-and-place policy is a privileged expert; its evaluation workflow and
reliability numbers are in
[research/scripted_experts](../research/scripted_experts/README.md).

### Validate the contracts

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/core/acceptance/so101/test_kinematics.py tests/core/acceptance/so101/test_scene.py tests/core/unit/test_robot_registry.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/core/integration/test_ros2_adapters.py::test_ros2_sim_teleop_command_moves_so101 -q
```

The second check covers the transport and needs no ROS2 installation.

### Run the real ROS2 node

Real message and executor coverage needs ROS2 Jazzy:

```bash
source /opt/ros/jazzy/setup.bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/core/robots/so101/test_so101_ros2_node.py -q
MUJOCO_GL=egl uv run python scripts/run_ros2_sim.py --manifest configs/manifests/so101_single_cube_fixed_place.yaml --seed 0 --max-steps 500
uv run python scripts/show_ros2_contract.py
```

The node accepts joint trajectory and gripper commands and publishes joint states, camera
frames, camera info and the SO-101 TF tree. The fake-transport test verifies command
translation and the real-node test verifies ROS2 message and executor behavior; neither
proves hardware connectivity or production QoS. The last command prints every shared
endpoint.

### Next steps and open work

- Imitation learning: [research/imitation_learning](../research/imitation_learning/README.md).
- Visual servoing baseline: [research/classical_control](../research/classical_control/README.md).
- Planner contract: `uv run python scripts/plan_task.py --help`. Only the scripted planner
  ships; a model-backed planner implements the same `Planner` contract, and its output
  must pass capability and safety checks before producing commands.
- FK, Jacobian and IK benchmark: `uv run python scripts/benchmark_ik.py --targets 20 --seed 0`
  (success rate, position and orientation error, iterations, runtimes).
- `SO101ROS2Node.handle_cartesian_target()` returns structured acceptance, IK error,
  iterations and a failure reason for a Cartesian request. It is not yet bound to a ROS2
  service or action endpoint.
- Main files: `configs/manifests/so101_single_cube_fixed_place.yaml`,
  `src/physai/robots/so101/mujoco_env.py`, `src/physai/sim/mujoco/scenes/common.py`,
  `research/scripted_experts/so101_pick_place_expert.py`.

## TurtleBot4

### Fetch, open and validate

```bash
uv run python scripts/fetch_assets.py --robot turtlebot4
uv run python scripts/run_sim.py --manifest configs/manifests/turtlebot4.yaml --serve --seed 0
uv run python scripts/run_sim.py --manifest configs/manifests/turtlebot4.yaml --seed 0 --max-steps 300
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/core/unit/test_robot_registry.py tests/core/acceptance/turtlebot/test_navigation.py -q
```

At yaw zero, positive linear velocity drives along the model's world `-Y` direction; the
navigation controller and tests use this convention. Use `--viewer` instead of `--serve`
for MuJoCo's desktop window. The tests cover registry creation, reset determinism, wheel
motion, ground contact and camera output.

### ROS2 node

```bash
source /opt/ros/jazzy/setup.bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/core/robots/turtlebot/test_turtlebot_ros2_node.py -q
uv run python scripts/run_ros2_sim.py --robot turtlebot4 --max-steps 100
```

The node subscribes to `/cmd_vel` and publishes `/joint_states`, `/odom` and the `odom` to
`base_link` transform.

### Deterministic RPP baseline

```bash
uv run python scripts/eval_navigation.py --robot turtlebot4 --goal-x 1.0 --goal-y -1.0 --goal-yaw 0.0 --seed 0 --max-steps 300
```

The output should include `reached=True` and `collisions=0`, and repeating the same seed
should give the same result. This is the robot-owned regulated pure-pursuit baseline.

### Nav2

The first integration uses a static map instead of SLAM:
`configs/maps/open_space/` (open 4 m by 4 m map with a border wall),
`configs/nav2/turtlebot4/params.yaml` (planner, controller, costmaps) and
`launch/nav2.launch.py` (driver, map server, TF, Nav2). Nav2 must be installed
(`ros2 pkg prefix nav2_bringup`); the repository Docker image has it:

```bash
docker compose -f docker/docker-compose.yml build physai
docker compose -f docker/docker-compose.yml run --rm physai ros2 pkg prefix nav2_bringup
ros2 launch launch/nav2.launch.py robot:=turtlebot4
uv run python scripts/send_nav_goal.py --robot turtlebot4 --x 1.0 --y 0.0 --yaw 0.0
```

Run `source /opt/ros/jazzy/setup.bash` in each terminal. The goal command succeeds only
when Nav2 returns `SUCCEEDED` and final odometry is within the position-error tolerance;
the validated baseline reaches the goal with about `0.244 m` final error under the default
goal checker. The launch uses an identity `map` to `odom` transform and validates map
loading, TF, planner startup, controller output and an open-world `NavigateToPose` goal.

For the obstacle scenario:

```bash
uv run python scripts/validate_nav2_obstacle.py
```

The runner starts and stops the whole Nav2 graph, waits for the lifecycle nodes, checks
that `/scan` sees the physical obstacle and sends the deterministic goal, exiting nonzero
if scan detection or goal acceptance fails. The equivalent manual launch is
`ros2 launch launch/nav2.launch.py robot:=turtlebot4 scenario:=obstacle_course map-file:=$PWD/configs/maps/obstacle_course/map.yaml`.
The scenario publishes a real `LaserScan` into the local obstacle layer and enables
Collision Monitor slowdown and stop zones; goal `(1.0, 0.0)` returned `SUCCEEDED` with
`0.021 m` final error, and the obstacle read `0.400 m` in `/scan`.

### Open work

Count physical MuJoCo contacts during the obstacle acceptance run (it currently checks
only `/scan` detection and the goal result). Keep the open map for the first smoke test
and move to a real map only once the map-frame and odometry-frame relationship is
understood. Planner, VLM and VLA stages can later target TurtleBot waypoints through the
same plan and action contracts.
