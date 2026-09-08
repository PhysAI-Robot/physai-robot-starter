# TurtleBot4 Runbook

This is the end-to-end TurtleBot4 workflow. Start with the MuJoCo base and
velocity control, then validate ROS2 odometry, then run the deterministic RPP
baseline before using Nav2.

## 1. Fetch and Open the Robot

```bash
uv run python scripts/fetch_assets.py --robot turtlebot4
```

Open the robot in MuJoCo:

```bash
uv run python scripts/run_sim.py \
  --robot turtlebot4 \
  --viewer \
  --seed 0
```

Run headless for repeatable checks:

```bash
uv run python scripts/run_sim.py \
  --robot turtlebot4 \
  --no-video \
  --seed 0 \
  --max-steps 300
```

At yaw zero, positive linear velocity drives along the model's world `-Y`
direction. The navigation controller and tests use this model convention.

## 2. Validate Basic Differential-Drive Control

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest \
  tests/test_robot_registry.py -q
```

These checks cover registry creation, reset determinism, wheel motion, ground
contact, and camera output.

## 3. Run the Real ROS2 Node

Real message and executor coverage requires ROS2 Jazzy:

```bash
source /opt/ros/jazzy/setup.bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest \
  tests/robots/turtlebot/test_ros2_node.py -q
```

Run a bounded ROS2 MuJoCo smoke test:

```bash
source /opt/ros/jazzy/setup.bash
uv run python scripts/run_ros2_sim.py \
  --robot turtlebot4 \
  --max-ticks 100
```

The node subscribes to `/cmd_vel` and publishes `/joint_states`, `/odom`, and
the `odom` to `base_link` TF transform.

## 4. Run the Deterministic RPP Baseline

Run the first Point A to Point B scenario directly in MuJoCo:

```bash
uv run python scripts/eval_turtlebot_navigation.py \
  --goal-x 1.0 \
  --goal-y -1.0 \
  --goal-yaw 0.0 \
  --seed 0 \
  --max-steps 300
```

Expected output includes `reached=True` and `collisions=0`. Repeat with the
same seed to verify deterministic results. This is the robot-owned regulated
pure-pursuit baseline.

## 5. Run Nav2 with the Dummy Map

The first Nav2 integration uses a standard static map instead of SLAM:

- `configs/nav2/dummy_map.yaml`: map metadata, resolution, and origin.
- `configs/nav2/dummy_map.pgm`: open 4 m by 4 m map with a border wall.
- `configs/nav2/nav2_params.yaml`: map, planner, controller, and costmaps.
- `configs/nav2/turtlebot4_nav2.launch.py`: driver, map server, TF, and Nav2.

Check that Nav2 is installed:

```bash
source /opt/ros/jazzy/setup.bash
ros2 pkg prefix nav2_bringup
```

The repository Docker image installs the required Nav2 packages. Build it
before running the launch when the host ROS2 installation does not contain
Nav2:

```bash
docker compose -f docker/docker-compose.yml build physai
docker compose -f docker/docker-compose.yml run --rm physai \
  ros2 pkg prefix nav2_bringup
```

Start the map-based smoke test when the package is available:

```bash
ros2 launch configs/nav2/turtlebot4_nav2.launch.py
```

In a second terminal, send the deterministic open-space goal:

```bash
source /opt/ros/jazzy/setup.bash
uv run python scripts/send_turtlebot_nav_goal.py \
  --x 1.0 \
  --y 0.0 \
  --yaw 0.0
```

The command succeeds only when Nav2 returns `SUCCEEDED` and final odometry is
within the configured position-error tolerance. The validated baseline reaches
the goal with approximately `0.244 m` final position error under the default
Nav2 goal checker.

The launch uses an identity `map` to `odom` transform and an open static map.
It validates map loading, TF connectivity, planner startup, controller output,
and an open-world `NavigateToPose` goal. It does not prove obstacle avoidance.

The current TurtleBot node does not publish `sensor_msgs/msg/LaserScan`, so the
local costmap intentionally has no obstacle layer yet.

## 6. Parameters and Open Work

Navigation parameters are in `configs/nav2/nav2_params.yaml`. Keep the dummy
map for the first Nav2 smoke test; replace it with a real map only after the
map-frame and odometry-frame relationship is understood.

Open navigation work is:

- Publish a real simulated `LaserScan`.
- Add obstacle geometry and obstacle-layer configuration.
- Add collision and timeout regression scenarios.
- Verify repeated Nav2 goals with the same seed.

Later planner/VLM/VLA stages can target TurtleBot waypoints through the same
shared plan and action contracts; they are not required for the first
navigation baseline.