# Robot runbooks

End-to-end workflows for each supported robot. The browser viewer's controls, recording and
troubleshooting are in the [Web Viewer runbook](WEB_VIEWER_RUNBOOK.md).

## SO-101

```bash
uv run python scripts/fetch_assets.py --robot so101
uv run python scripts/run_sim.py --serve --seed 0
uv run python scripts/run_sim.py --manifest configs/manifests/so101_single_cube_fixed_place.yaml --serve --seed 0
uv run python scripts/run_sim.py --manifest configs/manifests/so101_single_cube_fixed_place.yaml --seed 0 --max-steps 500
uv run python scripts/run_sim.py --manifest configs/manifests/so101_single_cube_randomized_place.yaml --serve --seed 0
uv run python scripts/run_sim.py --manifest configs/manifests/so101_stack_cubes.yaml --policy demo --serve --seed 0
```

The fetch includes the upstream `so101_new_calib_camera.xml` variant and its wrist camera mount
meshes; the simulator selects that file when present. The first command opens an idle robot in
the browser, the second the checked-in pick-and-place scene, the third runs it headless
(`--video` writes frames under `outputs/`). MuJoCo's own desktop viewer (`--viewer`, needs a
display) is a fallback. The default `demo` policy is a privileged open-loop pick-and-place
(true object poses, `ScriptedPlanner` + `PlanRunner`): a proof that a policy drives the arm, not
a baseline. Your own policies and scenes come from a plugin named in the manifest
([README](../README.md#bring-your-own-research)); this project's own expert, visual servo and
ACT workflows are in `research/` in [`physai-research`](https://github.com/PhysAI-Robot/physai-research).

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

**Engines.** `--sim mujoco|isaac|newton` picks the physics and `--render mujoco|isaac|newton` the
camera renderer (default: the simulator's own); every pairing runs the same manifest, for example
`uv run python scripts/run_sim.py --sim newton --render isaac --policy demo --video`
([decision I](DECISIONS.md#i-three-engines-and-the-renderer-is-chosen-apart-from-the-physics)).
`demo` runs on MuJoCo and Newton; on Isaac use an observation-only policy such as `constant`.
Newton is installed by `uv sync` and needs a GPU; its first step compiles kernels (a few minutes
the first time, cached after). Validate with
`uv run python -m pytest tests/core/acceptance/so101/test_newton_env.py -q`.

`--video` records the scene's diagonal `video` view (the whole arm and the table) with whichever
renderer you chose; `--camera front|wrist` records an observation camera instead. A video ends in
`_o` for a success and `_x` for a failure.

Other entry points: FK/Jacobian/IK benchmark `uv run python scripts/benchmark_ik.py --targets 20
--seed 0`; planner contract `uv run python scripts/plan_task.py --manifest <session> --seed 3` (SO-101 and ViperX 300 S; only the scripted
planner ships); this project's learning and vision baselines are plugins in
[`physai-research`](https://github.com/PhysAI-Robot/physai-research).

## ViperX 300 S (ALOHA follower arm)

One six-joint ViperX 300 S arm from MuJoCo Menagerie (`trossen_vx300s`, BSD-3), a
front camera and a D405-style wrist camera. It is the second manipulator for the same
single-cube pick-and-place task, so the web viewer, the recorder, the task runtime and
`eval_policy.py` work on it unchanged.

```bash
uv run python scripts/fetch_assets.py --robot viperx300s
uv run python scripts/run_sim.py --manifest configs/manifests/viperx300s_single_cube_fixed_place.yaml --policy constant --serve --seed 0
uv run python scripts/run_sim.py --manifest configs/manifests/viperx300s_single_cube_randomized_place.yaml --policy constant --serve --seed 0
uv run python scripts/eval_policy.py --manifest research/configs/viperx300s_single_cube_randomized_place.yaml --policy scripted --episodes 100 --seed 1000
```

The first command is the idle robot in the browser with both cameras; the `scripted` expert is a
research policy (`research/scripted_experts/viperx300s_pick_place_expert.py`, 100/100 on seeds
1000-1099). Validate with `uv run python -m pytest tests/core/acceptance/viperx300s -q`.
The core `ScriptedPlanner` with `PlanRunner` also places the cube with this arm's `GraspProfile`
(`scripts/plan_task.py --manifest configs/manifests/viperx300s_single_cube_fixed_place.yaml`), but core
ships no ViperX policy.

What this is not, so results are not over-read:

- **The gripper is the stock ViperX two-finger gripper, not the ALOHA 2 follower's custom
  gripper.** Menagerie's single-arm model has the stock one; the custom gripper exists only in
  its bimanual `aloha` model. The arm's kinematics and joint limits match the ALOHA follower; the
  fingers do not.
- The gripper joint is a finger position in **metres** (0.021 closed, 0.057 open), while the core
  declares joint positions in radians. The robot's `gripper_unit` metadata says so; commands use the
  usual normalised 0 (closed) .. 1 (open).
- Gravity is cancelled on the arm's bodies by default (`EnvConfig.gravity_compensation`). The
  Menagerie actuators are soft and sag by centimetres under the arm's weight; the real servos hold
  their position. Turn it off to see the sag.
- **Isaac Sim.** There is no URDF, so `--sim isaac` imports the Menagerie MJCF with Isaac's own MJCF
  importer (`fetch_assets.py --robot viperx300s` writes the USD to `assets/viperx300s/usd/` when Isaac Sim is installed; a run reads it and stops with that command if it is missing).
  On the maintainer's RTX 3060 the arm holds its home pose, both fingers close and reopen, and the
  per-episode render check (`robot_is_rendered`) passes: Isaac draws the arm, both fingers and the cube
  in the front and `video` views, from its own state and, with `--render isaac`, from MuJoCo's or
  Newton's. An earlier session in which no arm was drawn did not recur. The scripted expert is
  MuJoCo-only (it runs on Newton, and Isaac draws its episodes), so the research pick and place has not
  been run on Isaac physics for this arm. Core's scripted planner does grasp, lift and hold the cube there
  (`tests/core/acceptance/test_held_cube.py`: the cube moves 0.00 mm in 60 s).
  The importer does not read the position actuators' `kp`, so the drive gains in
  `robots/viperx300s/description.yaml` are transcribed from the MJCF; PhysX has no equality constraint,
  so the environment drives both fingers itself.
- **Newton** (`--sim newton`, `--render newton`) follows the MuJoCo environment (same MJCF, gravity
  compensation and equality-constrained fingers); the scripted expert succeeds on it
  (`uv run python -m pytest tests/core/acceptance/viperx300s/test_viperx300s_newton.py -q`). With
  `--sim mujoco|newton --render isaac` Isaac draws the arm and both fingers from the other engine's state.
- The wrist camera pose is the D405 mount of the Menagerie `aloha` model expressed on this arm's
  gripper link; it is a plausible mount, not a calibrated one.

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
