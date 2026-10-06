# Robot runbooks

End-to-end workflows for each supported robot. The browser viewer's controls, recording and
troubleshooting are in the [Web Viewer runbook](WEB_VIEWER_RUNBOOK.md).

## SO-101

```bash
uv run python scripts/fetch_assets.py --robot so101
uv run python scripts/run_sim.py --serve --seed 0
uv run python scripts/run_sim.py --manifest configs/manifests/so101_single_cube_fixed_place.yaml --serve --seed 0
uv run python scripts/run_sim.py --manifest configs/manifests/so101_single_cube_fixed_place.yaml --seed 0 --max-steps 500
```

The fetch includes the upstream `so101_new_calib_camera.xml` variant and its wrist camera mount
meshes; the simulator selects that file when present. The first command opens an idle robot in
the browser, the second the checked-in pick-and-place scene, the third runs it headless
(`--video` writes frames under `outputs/`). MuJoCo's own desktop viewer (`--viewer`, needs a
display) is a fallback. The scripted pick-and-place policy is a privileged expert; its
evaluation workflow and numbers are in
[research/scripted_experts](../research/scripted_experts/README.md).

Validate the contracts (the second check covers the ROS2 transport and needs no ROS2):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/core/acceptance/so101/test_kinematics.py tests/core/acceptance/so101/test_scene.py tests/core/unit/test_robot_registry.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/core/integration/test_ros2_adapters.py::test_ros2_sim_teleop_command_moves_so101 -q
```

The real node needs ROS2 Jazzy. It accepts joint trajectory and gripper commands and
publishes joint states, camera frames, camera info and the SO-101 TF tree; neither the
fake-transport test nor the real-node test proves hardware connectivity or production QoS:

```bash
source /opt/ros/jazzy/setup.bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/core/robots/so101/test_so101_ros2_node.py -q
MUJOCO_GL=egl uv run python scripts/run_ros2_sim.py --manifest configs/manifests/so101_single_cube_fixed_place.yaml --seed 0 --max-steps 500
uv run python scripts/show_ros2_contract.py    # prints every shared endpoint
```

Other entry points: FK/Jacobian/IK benchmark `uv run python scripts/benchmark_ik.py --targets 20
--seed 0`; planner contract `uv run python scripts/plan_task.py --help` (only the scripted
planner ships); learning and vision baselines in
[research/imitation_learning](../research/imitation_learning/README.md) and
[research/classical_control](../research/classical_control/README.md).

## TurtleBot4

TurtleBot4 is maintained, not developed further.

```bash
uv run python scripts/fetch_assets.py --robot turtlebot4
uv run python scripts/run_sim.py --manifest configs/manifests/turtlebot4.yaml --serve --seed 0
uv run python scripts/run_sim.py --manifest configs/manifests/turtlebot4.yaml --seed 0 --max-steps 300
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/core/unit/test_robot_registry.py tests/core/acceptance/turtlebot/test_navigation.py -q
```

At yaw zero, positive linear velocity drives along the model's world `-Y` direction; the
navigation controller and tests use this convention.

**ROS2 node.** It subscribes to `/cmd_vel` and publishes `/joint_states`, `/odom` and the
`odom` to `base_link` transform:

```bash
source /opt/ros/jazzy/setup.bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/core/robots/turtlebot/test_turtlebot_ros2_node.py -q
uv run python scripts/run_ros2_sim.py --robot turtlebot4 --max-steps 100
```

**Deterministic baseline** (the robot-owned regulated pure-pursuit controller; the output
should include `reached=True` and `collisions=0`, and repeat for the same seed):

```bash
uv run python scripts/eval_navigation.py --robot turtlebot4 --goal-x 1.0 --goal-y -1.0 --goal-yaw 0.0 --seed 0 --max-steps 300
```

**Nav2.** The integration uses a static map: `configs/maps/open_space/`,
`configs/nav2/turtlebot4/params.yaml` and `launch/nav2.launch.py`. Nav2 must be installed
(the Docker image has it); run `source /opt/ros/jazzy/setup.bash` in each terminal:

```bash
ros2 launch launch/nav2.launch.py robot:=turtlebot4
uv run python scripts/send_nav_goal.py --robot turtlebot4 --x 1.0 --y 0.0 --yaw 0.0
uv run python scripts/validate_nav2_obstacle.py   # whole obstacle scenario, nonzero exit on failure
```

The goal command succeeds only when Nav2 returns `SUCCEEDED` and the final odometry is within
the position-error tolerance. The obstacle scenario
(`ros2 launch launch/nav2.launch.py robot:=turtlebot4 scenario:=obstacle_course map-file:=$PWD/configs/maps/obstacle_course/map.yaml`)
publishes a real `LaserScan` into the local obstacle layer and enables Collision Monitor
slowdown and stop zones. Not yet covered: counting physical MuJoCo contacts during the
obstacle run (it checks only `/scan` detection and the goal result).
