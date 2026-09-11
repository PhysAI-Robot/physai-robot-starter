# 🗺️ Project Roadmap: physai-robot-starter

`physai-robot-starter` is an open-source starter kit for Embodied AI & Robotics. It bridges classical ROS 2 control stacks with visual control, learning-based motor skills, high-level vision-language planning, and end-to-end vision-language-action policies in MuJoCo.


```

[Phase 1: Classical Foundation & ROS 2] ➔ [Phase 2: Learning-Based Motor Skills] ➔ [Phase 3: VLM Orchestration] ➔ [Phase 4: End-to-End VLA]

```

## Current Status

**Current phase: Phase 1 - Classical Foundation & ROS 2 Contract.**

The MuJoCo baseline, robot registry, capability-aware action/observation
contracts, SO-101 kinematics, TurtleBot4 model, and initial ROS 2-shaped
contracts are already in place. The active work is to make the Phase 1
foundation reliable and executable through deterministic tests, robot control
adapters, and the first ROS 2 integration.

Phase 2 and later are future direction only. They should consume the stable
contracts produced by Phase 1, not drive changes to those contracts ad hoc.

The checklist below tracks implementation evidence in the repository, not
phase completion. `[x]` means the deliverable exists and has focused coverage;
`[ ]` means it is planned, missing, or only partially implemented. A phase is
complete only when its Definition of Done also passes.

---

## 🏗️ Phase 1: Classical Robotics Foundation & ROS 2 Contract

### Objective
Make the existing SO-101 and TurtleBot4 MuJoCo implementations reliable
through stable contracts, deterministic control, and a first ROS 2 integration.
Keep the architecture extensible for the Standalone Franka Panda and Google
Mobile Manipulator, but do not make those two robots requirements for the first
Phase 1 completion gate.

The Phase 1 implementation should preserve the current capability-aware design:
fixed-base manipulators expose joint and gripper capabilities, mobile bases
expose base velocity and odometry, and a future mobile manipulator may combine
both. Do not force every embodiment into one identical action array.

### Phase 1A: Contracts and Deterministic Simulator Baseline

Stabilize the interfaces that every later controller, ROS 2 node, and learning
policy will consume.

#### Deliverables
- [x] `configs/sim_config.yaml`: Centralized simulation configuration with `domain_randomization.enabled: false` by default.
- [x] `src/physai/robots/registry.py`: Capability-aware robot discovery and factory API for the currently supported robots.
- [x] Contract validation for action modes, joint names, camera names, timestamps, frame IDs, shapes, and finite values.
- [x] Explicit runtime validation for declared units such as radians, metres, and metres per second through `RobotSpec` unit declarations.
- [x] Deterministic reset and seed handling for SO-101 and TurtleBot4.
- [x] Smoke and regression tests covering `reset()`, `step()`, action validation, and capability requirements.

#### Definition of Done
- [x] `available_robots()` reports SO-101 and TurtleBot4 without importing optional ROS 2 or ML dependencies.
- [x] Repeating an episode with the same seed produces the same initial state and task randomization, including the three-cube sorting reset regression.
- [x] Invalid action modes, shapes, joint orders, and unsupported capabilities fail with clear errors.
- [x] The existing scripted SO-101 workflow and TurtleBot4 twist workflow remain runnable after contract changes. **Verified:** the scripted SO-101 task completes `20/20` in the deterministic reliability check (`--episodes 20 --seed 0 --max-steps 600`). The result required fixing an unreachable `lift_height` (IK never converged, freezing the arm), calibrating `gripper_grip` and `gripper_force_limit`, placing the added pad geoms on the actual jaw contact surfaces, and disabling the original jaw collision meshes so contacts are not duplicated.

### Phase 1B: SO-101 Control and ROS 2 Bridge

Deliver the first complete ROS 2 control path for the robot with the most
complete task and kinematics support.

#### Deliverables
- [x] `src/physai/bridge/mujoco_ros_bridge.py`: Synchronous runtime MuJoCo bridge that publishes joint states and camera frames and accepts joint trajectory and gripper commands through an injected transport.
- [x] ROS 2 message adapters for `sensor_msgs/msg/JointState`, `sensor_msgs/msg/Image`, `trajectory_msgs/msg/JointTrajectory`, and the gripper command interface, with real `rclpy` node acceptance coverage.
- [x] TF publication for the documented SO-101 frame tree, including camera and gripper frames.
- [x] Teleoperation path through a real `rclpy` node and ROS2 joint and gripper topics.
- [x] Integration test for command-to-simulation and simulation-to-topic flow using the transport port and fake ROS 2 transport.

#### Definition of Done
- [x] A joint trajectory command moves the SO-101 in MuJoCo at the configured control rate through a real `rclpy` node.
- [x] Published joint names, radians, timestamps, camera encoding, and frame IDs match the ROS 2 contract for the fields currently represented by `Observation`.
- [x] Gripper commands are converted consistently between normalized aperture and simulator joint units.
- [x] The bridge can run with rendering disabled and does not require ML packages.

### Phase 1C: TurtleBot4 Navigation Foundation

Add navigation only for the mobile-base embodiment. SO-101 does not need Nav2.
Start with a small deterministic world and a simple controller before adding
more complex planners.

#### Deliverables
- [x] TurtleBot4 ROS 2 MuJoCo bridge for `/cmd_vel`, wheel state, `/odom`, and TF.
- [x] `configs/nav2/turtlebot4/params.yaml`: TurtleBot4 Nav2 controller, LaserScan obstacle layer, inflation, and Collision Monitor parameters.
- [x] Direct MuJoCo Point A to Point B scenario with known start and goal poses.
- [x] TurtleBot-owned RPP controller as the initial direct-simulation baseline; evaluate the Nav2 controller separately when Nav2 is installed.
- [x] Deterministic TurtleBot4 obstacle scenario with a physical box, static map, LaserScan, and Collision Monitor.
- [x] Automated obstacle-navigation acceptance runner in `scripts/validate_nav2_obstacle.py`; it validates lifecycle startup, scan detection, and `NavigateToPose` success.

#### Definition of Done
- [x] TurtleBot4 accepts a standard `geometry_msgs/msg/Twist` command and reports wheel state, odometry, and `odom` to `base_link` TF through the real `rclpy` acceptance test.
- [x] Nav2 reaches a goal in the deterministic test world without collision. The obstacle acceptance path checks `NavigateToPose` success, final position error, and the MuJoCo non-ground collision count; the latest direct acceptance result was `SUCCEEDED` with `0.041 m` final error and zero collisions.
- [x] The direct MuJoCo navigation result is reproducible across repeated runs with the same seed.
- [x] Direct and Nav2 navigation failures report structured action status, timeout reason, final position error, and collision count; `scripts/send_nav_goal.py` can write a JSON report.

### Phase 1D: Per-Robot Kinematics and Manipulation Control

Keep kinematics implementations beside the robot they describe. The current
SO-101 implementation is numerical damped-least-squares IK, so analytical IK
should not be a Phase 1 requirement unless a later robot specifically needs it.

#### Deliverables
- [x] Benchmark the existing SO-101 FK, Jacobian, and numerical IK over a defined set of reachable targets with `scripts/benchmark_ik.py`.
- [x] Expose Cartesian targeting through the transport-neutral ROS 2 service/action contract in `src/physai/bridge/cartesian.py`; `SO101ROS2Node.handle_cartesian_target()` resolves pose requests through IK and safety validation.
- [x] Validate reachable-target position error, convergence, joint limits, and gripper contact behavior in simulator tests.
- [x] Complete orientation-error and collision/contact acceptance coverage with recorded simulator metrics; a broader kinematics benchmark remains open.
- [ ] Add a separate kinematics adapter for each future arm embodiment instead of generalizing SO-101 assumptions. **Foundation:** `src/physai/robots/kinematics_registry.py` now provides the registration boundary; concrete Franka and mobile-manipulator adapters remain future work.

#### Definition of Done
- [x] SO-101 IK reaches the documented test targets within the configured position and orientation tolerances.
- [x] Unreachable targets fail explicitly and do not emit unsafe joint targets.
- [x] Joint-limit and collision checks are included in the acceptance test, not only final end-effector position.
- [x] The benchmark records success rate, position/orientation error, iterations, FK/Jacobian/IK runtime, and Jacobian shape.

### Phase 1E: Controlled Domain Randomization

Add randomization only after the deterministic controller and ROS 2 paths are
stable. Keep all randomization behind one configuration and seed so failures
remain reproducible.

#### Deliverables
- [x] `src/physai/sim/domain_randomization.py`: Seeded engine for selected physics, visual, camera, and optional clutter parameters.
- [x] Configuration for friction, mass, lighting, camera pose, clutter ranges, and protected-task-point clearance with documented defaults.
- [x] Explicit `enabled: false` behavior that preserves the deterministic baseline.
- [x] Seeded randomization metadata recorded in episode and evaluation output.
- [x] Regression comparison between deterministic and randomized runs in `scripts/eval_randomization.py`.

#### Definition of Done
- [x] Switching `domain_randomization.enabled` between `false` and `true` does not change ROS 2 topic names or message schemas.
- [x] The same seed reproduces the same randomized parameters.
- [x] Randomized values stay within documented safe ranges and clutter placement rejects task-critical positions instead of silently invalidating the scene.
- [x] Baseline control success and failure rates are reported separately for deterministic and randomized settings.

### Phase 1 Scope Boundary

The first Phase 1 completion gate covers **SO-101 + TurtleBot4**. Standalone
Franka Panda and Google Mobile Manipulator remain planned embodiments and can
be added after the shared contracts, bridge pattern, and acceptance tests are
proven on the first two robots.

### Definition of Done (DoD)
- [x] SO-101 and TurtleBot4 pass the deterministic contract, reset, and control regression suite. The current suite has 92 passing tests and 2 skipped in the ROS2 Jazzy environment; the scripted SO-101 pick-and-place reliability check is 20/20 with the calibrated pad setup.
- [x] SO-101 can be teleoperated through a real `rclpy` node using its ROS 2 joint, gripper, camera, and TF interfaces.
- [x] TurtleBot4 can navigate from Point A to Point B through the ROS 2/Nav2 path without collision in the deterministic test world. Automated acceptance validates LaserScan obstacle detection, Collision Monitor, final pose error, and zero MuJoCo contact count.
- [x] SO-101 IK meets the documented position and orientation tolerances on reachable targets and rejects invalid targets safely; broader runtime benchmarking remains open.
- [x] Domain Randomization can be enabled or disabled through `configs/sim_config.yaml` without changing ROS 2 topic contracts, with seeded metadata and deterministic-vs-randomized evaluation coverage.
- [x] The bridge and simulator can run without Phase 2+ dependencies such as LeRobot, VLM, or VLA packages.

---

## Phase 1 to Phase 2 Bridge: Training Readiness

Add the smallest training boundary on top of the completed Phase 1 contracts.
This work must adapt the existing `Observation -> Action` interface for
learning tools without moving training logic into robot environments or ROS 2
adapters.

### Bridging Deliverables
- [ ] Define canonical `ObservationSpec` and `ActionSpec` schemas covering names, shapes, dtypes, units, ranges, camera layout, and normalization metadata.
- [ ] Add a Gymnasium-compatible environment adapter for direct MuJoCo task training with seeded `reset()`, `step()`, spaces, render modes, and structured episode information.
- [ ] Make the canonical SO-101 action layout explicit and consistent across policies, recorder output, replay, ROS 2 conversion, and training datasets.
- [ ] Version dataset metadata with robot, task, contract schema, simulator configuration, camera configuration, seed, and train/validation/test split information.
- [ ] Add checkpoint metadata and compatibility validation for robot, task, observation schema, action schema, normalization, and training configuration.
- [ ] Add shared evaluation reports for success, collision, timeout, unsafe action, reward, and held-out seed performance.
- [ ] Add a smoke test that runs one episode through the training adapter and confirms that the resulting action still passes the existing safety and robot validation gates.

### Bridge Completion Gate
The Phase 2A visual-servoing baseline and all learned policies must consume the
same canonical observation and action schemas. A training framework may be
changed later, but policies must remain evaluable through the repository's
policy boundary without changing the robot adapter or ROS 2 contract.

---

## 🎯 Phase 2: Learning-Based Motor Skills

### Objective
Build low-level motor skills on top of the stable Phase 1 contracts. Establish a
classical visual-control baseline first, then add learned policies through
imitation learning and deep reinforcement learning. All three tracks must
produce the same `Observation -> Action` policy interface.

### Phase 2A: Visual Servoing Baselines

Implement explicit camera-feedback controllers before training learned visual
policies. The first targets are SO-101 wrist-camera alignment and object
approach, followed by TurtleBot4 visual goal tracking where a suitable visual
target is available.

#### Key Deliverables
- [ ] Visual feature or fiducial detection with documented camera-frame and robot-frame transforms.
- [ ] Image-based or pose-based visual servo controller producing bounded `Action` values through the existing safety layer.
- [ ] SO-101 visual alignment and approach acceptance test with position error, settling time, and failure reporting.
- [ ] Deterministic replay and robustness evaluation under bounded camera and scene perturbations.

#### Definition of Done
1. A visual servo controller reaches and holds a documented image or pose target from multiple initial conditions.
2. Commands respect the existing joint limits, action-step limits, and collision checks.
3. Evaluation reports visual error, end-effector error, settling time, and failure reason separately from learned-policy metrics.

### Phase 2B: Vision-Based Motor Skills (Imitation Learning via LeRobot)

Transition from explicit visual control to data-driven policies. Use Hugging
Face's **LeRobot** ecosystem to train local policies that map raw camera
observations and robot state to low-level motor actions. The Phase 2A
controller remains the interpretable baseline for comparison.

#### Key Deliverables & Directory Layout
- [ ] `scripts/collect_demos.py`: Automated trajectory recorder producing a LeRobot-shaped `.npz` and metadata layout for the SO-101. **Partial:** it does not yet export the standard `LeRobotDataset` format or support teleoperation/base odometry.
- [ ] `docker/Dockerfile.lerobot`: Headless containerized environment for policy training and dependency isolation.
- [ ] `scripts/train_policy.py`: Local GPU or cloud training pipeline for ACT (Action Chunking with Transformers) or Diffusion Policy models. **Partial:** the ACT-specific `scripts/train_act.py` exists; the planned unified entry point does not.
- [ ] `scripts/eval_policy.py`: Closed-loop evaluation runner executing policies over the direct MuJoCo interface. **Partial:** scripted, replay, and LeRobot policies are supported; ROS 2 evaluation is not integrated.

#### Interface Contract
* **Planned input observation:** `camera_wrist_rgb` ($224 \times 224$), `camera_top_rgb` ($224 \times 224$), `joint_states`
* **Planned output action:** Predicted joint position target sequences / base action vectors ($N=100$ chunking horizon).
* **Current prototype:** SO-101 `front` and `wrist` RGB frames, six-value joint state, and configurable ACT chunks (default `30`) in `src/physai/policy/act_dataset.py`.

#### Definition of Done (DoD)
1. Harvest 50–100 successful task episodes per manipulation skill exported cleanly to `LeRobotDataset`.
2. Policy training finishes with converging loss curves inside the containerized environment.
3. Closed-loop evaluation in `eval_policy.py` achieves **> 80% task success rate** over 20 randomized trials in MuJoCo, bypassing analytical solvers.

### Phase 2C: Deep Reinforcement Learning

Train policies directly against MuJoCo task rewards after the observation,
action, safety, and evaluation paths are proven by the preceding baselines.
Start with state-based control so reward design and dynamics can be debugged
before adding pixel observations.

#### Key Deliverables
- [ ] `scripts/train_rl.py`: Reproducible PPO or SAC training entry point with checkpoint and metrics output.
- [ ] `scripts/eval_rl.py` or equivalent policy integration in `scripts/eval_policy.py` for held-out seeds.
- [ ] RL environment adapter exposing the existing `Observation`, `Action`, `RobotSpec`, and safety contracts.
- [ ] SO-101 state-based pick-and-place or TurtleBot4 navigation benchmark before vision-based RL.
- [ ] Seeded training, checkpoint metadata, deterministic evaluation, and domain-randomized evaluation reports.

#### Definition of Done
1. A PPO or SAC policy trains reproducibly from a fixed seed in MuJoCo.
2. A trained checkpoint runs through the existing policy boundary without changing the robot adapter.
3. The policy achieves at least 80% success over 20 held-out seeds on a documented task.
4. Randomized evaluation reports success, collision, timeout, and unsafe-action failures separately.
5. Vision-based RL is added only after the state-based benchmark passes.

---

## 🧠 Phase 3: High-Level Visual-Semantic Planning (VLM Integration)

### Objective
Introduce natural language human interfaces (text or speech). Implement a Large Vision-Language Model (**SmolVLM**) as a high-level reasoning orchestrator that decomposes abstract user commands and sequences the localized motor skills built in Phase 1 and Phase 2.

### Key Deliverables & Directory Layout
- [x] High-level planner contract and registry in `src/physai/planner/`, with scripted, SmolVLM, and Claude backends returning the shared `Plan`/`SubGoal` shape.
- [ ] `src/physai/audio/speech_to_text.py`: Whisper API / local Whisper integration enabling spoken voice instructions.
- [ ] `src/physai/vlm/plan_task.py`: SmolVLM reasoning node parsing dual-camera frames alongside natural language prompts. **Partial:** `scripts/plan_task.py` and `src/physai/planner/smolvlm.py` provide the local in-process workflow; the planned ROS2 VLM node path is absent.
- [ ] `src/physai/vlm/schemas.py`: Pydantic JSON schema enforcing structured action outputs from VLM responses. **Partial:** Claude uses a structured JSON-schema dictionary; no Pydantic schema module exists.
- [ ] `src/physai/vlm/error_recovery.py`: Closed-loop monitoring logic triggering VLM re-evaluations upon execution failure states.

### VLM Execution Schema Contract (Example)
```json
{
  "plan_id": "task_001",
  "subgoals": [
    {
      "step": 1,
      "action": "navigate_to",
      "target": [1.5, 0.2, 0.0]
    },
    {
      "step": 2,
      "action": "execute_learned_grasp",
      "target_label": "red_cube"
    }
  ]
}

```

### Definition of Done (DoD)

1. System correctly interprets abstract commands (e.g., *"Fetch the red cube from the desk and move to the shelf"*) into valid JSON action sequences.
2. High-level orchestrator successfully dispatches actions to ROS 2 Nav2 (Phase 1) or LeRobot Policy Action Servers (Phase 2).
3. If a grasp fails, `error_recovery.py` captures the updated camera frame, detects the failure, and issues a valid semantic fallback plan.

---

## ⚡ Phase 4: End-to-End Vision-Language-Action (VLA Policy)

### Objective

Eliminate modular boundaries between high-level planning and low-level execution. Build an automated data generation and training pipeline that fine-tunes a single unified VLA neural network to output raw joint/wheel velocities directly from text prompts and visual pixel streams.

### Key Deliverables & Directory Layout

* [ ] `docker/parallel_data_farm/`: Scalable, multi-container Docker cluster framework inside `docker/container.py` capable of farming thousands of randomized multi-task demonstration trajectories simultaneously in headless mode.
* [ ] `scripts/train_vla.py`: Fine-tuning pipeline for lightweight edge-optimized VLA model variants.
* [ ] `scripts/benchmark_robustness.py`: Rigorous stress-testing suite evaluating policy generalization under heavy Domain Randomization.

### Definition of Done (DoD)

1. Parallel data infrastructure successfully generates > 1,000 multi-task trajectories across headless MuJoCo containers.
2. Fine-tuned VLA policy controls the robot directly from raw pixels + text instructions at high frequency ($\ge 10\text{ Hz}$).
3. VLA policy exhibits zero-shot generalization under extreme Domain Randomization (swapping floor textures, dynamic lighting changes, background clutter, and surface friction noise with `domain_randomization.enabled: true`).

---

## 🔮 Future Expansion (Optional)

* **Cross-Simulator Portability (Sim2Sim):** Export trained policy checkpoints and unified robot API wrappers to **Isaac Lab** for ultra-large-scale parallel synthetic data generation or high-fidelity RTX rendering tests.
