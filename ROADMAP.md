# 🗺️ Project Roadmap: physai-robot-starter

`physai-robot-starter` is an open-source starter kit for Embodied AI & Robotics. It bridges classical ROS 2 control stacks with modern Data-Driven Motor Skills (LeRobot), High-Level VLM Planning (SmolVLM), and End-to-End VLA policies in MuJoCo.


```

[Phase 1: Classical Foundation & ROS 2] ➔ [Phase 2: Vision Skills (LeRobot)] ➔ [Phase 3: VLM Orchestration] ➔ [Phase 4: End-to-End VLA]

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
- [ ] Explicit runtime validation for declared units such as radians, metres, and metres per second.
- [x] Deterministic reset and seed handling for SO-101 and TurtleBot4.
- [x] Smoke and regression tests covering `reset()`, `step()`, action validation, and capability requirements.

#### Definition of Done
- [x] `available_robots()` reports SO-101 and TurtleBot4 without importing optional ROS 2 or ML dependencies.
- [ ] Repeating an episode with the same seed produces the same initial state and task randomization. **Partial:** baseline reset behavior is covered; sorting randomization still needs a dedicated regression assertion.
- [x] Invalid action modes, shapes, joint orders, and unsupported capabilities fail with clear errors.
- [ ] The existing scripted SO-101 workflow and TurtleBot4 twist workflow remain runnable after contract changes. **Partial:** both commands run, but the scripted SO-101 task currently completes with `0/5` success in the reliability check.

### Phase 1B: SO-101 Control and ROS 2 Bridge

Deliver the first complete ROS 2 control path for the robot with the most
complete task and kinematics support.

#### Deliverables
- [x] `src/physai/bridge/mujoco_ros_bridge.py`: Synchronous runtime MuJoCo bridge that publishes joint states and camera frames and accepts joint trajectory and gripper commands through an injected transport.
- [ ] ROS 2 message adapters for `sensor_msgs/msg/JointState`, `sensor_msgs/msg/Image`, `trajectory_msgs/msg/JointTrajectory`, and the gripper command interface. **Partial:** codec and `RclpyTransport` adapters exist; a real ROS 2 node integration is still pending.
- [ ] TF publication for the documented SO-101 frame tree.
- [ ] Teleoperation path using a standard joint controller or equivalent test client.
- [x] Integration test for command-to-simulation and simulation-to-topic flow using the transport port and fake ROS 2 transport.

#### Definition of Done
- [ ] A joint trajectory command moves the SO-101 in MuJoCo at the configured control rate. **Partial:** the bridge tick and command routing are tested with a fake robot; an end-to-end SO-101 ROS 2 test is still pending.
- [x] Published joint names, radians, timestamps, camera encoding, and frame IDs match the ROS 2 contract for the fields currently represented by `Observation`.
- [x] Gripper commands are converted consistently between normalized aperture and simulator joint units.
- [x] The bridge can run with rendering disabled and does not require ML packages.

### Phase 1C: TurtleBot4 Navigation Foundation

Add navigation only for the mobile-base embodiment. SO-101 does not need Nav2.
Start with a small deterministic world and a simple controller before adding
more complex planners.

#### Deliverables
- [ ] TurtleBot4 ROS 2 bridge for `/cmd_vel`, wheel state, `/odom`, and TF.
- [ ] `configs/nav2/`: Minimal Nav2 configuration and launch assets for the TurtleBot4 test world.
- [ ] Waypoint or Point A to Point B scenario with known start and goal poses.
- [ ] RPP controller as the initial baseline; evaluate MPPI separately if the simulator timing supports it.
- [ ] Obstacle and collision regression scenarios.

#### Definition of Done
- [ ] TurtleBot4 accepts a standard `geometry_msgs/msg/Twist` command and reports consistent odometry.
- [ ] Nav2 reaches a goal in the deterministic test world without collision.
- [ ] The result is reproducible across repeated runs with the same seed.
- [ ] Navigation failures report useful termination and timeout information.

### Phase 1D: Per-Robot Kinematics and Manipulation Control

Keep kinematics implementations beside the robot they describe. The current
SO-101 implementation is numerical damped-least-squares IK, so analytical IK
should not be a Phase 1 requirement unless a later robot specifically needs it.

#### Deliverables
- [ ] Benchmark the existing SO-101 FK, Jacobian, and numerical IK over a defined set of reachable targets.
- [ ] Expose Cartesian targeting through a ROS 2 service or action after the local IK behavior is validated.
- [x] Validate reachable-target position error, convergence, joint limits, and gripper contact behavior in simulator tests.
- [ ] Complete orientation-error and collision/contact acceptance coverage with recorded benchmark metrics.
- [ ] Add a separate kinematics adapter for each future arm embodiment instead of generalizing SO-101 assumptions.

#### Definition of Done
- [ ] SO-101 IK reaches the documented test targets within the configured position and orientation tolerances.
- [ ] Unreachable targets fail explicitly and do not emit unsafe joint targets.
- [ ] Joint-limit and collision checks are included in the acceptance test, not only final end-effector position.
- [ ] The benchmark records success rate, error, iterations, and runtime.

### Phase 1E: Controlled Domain Randomization

Add randomization only after the deterministic controller and ROS 2 paths are
stable. Keep all randomization behind one configuration and seed so failures
remain reproducible.

#### Deliverables
- [ ] `src/physai/sim/domain_randomization.py`: Engine for selected physics and visual parameters.
- [ ] Configuration for friction, mass, lighting, camera pose, and clutter ranges with documented defaults.
- [x] Explicit `enabled: false` behavior that preserves the deterministic baseline.
- [ ] Seeded randomization metadata recorded in episode or evaluation output.
- [ ] Regression comparison between deterministic and randomized runs.

#### Definition of Done
- [ ] Switching `domain_randomization.enabled` between `false` and `true` does not change ROS 2 topic names or message schemas.
- [ ] The same seed reproduces the same randomized parameters.
- [ ] Randomized values stay within documented safe ranges and do not silently invalidate robot models.
- [ ] Baseline control success and failure rates are reported separately for deterministic and randomized settings.

### Phase 1 Scope Boundary

The first Phase 1 completion gate covers **SO-101 + TurtleBot4**. Standalone
Franka Panda and Google Mobile Manipulator remain planned embodiments and can
be added after the shared contracts, bridge pattern, and acceptance tests are
proven on the first two robots.

### Definition of Done (DoD)
- [x] SO-101 and TurtleBot4 pass the deterministic contract, reset, and control regression suite. The current suite has 65 passing tests; task-level scripted success still needs stabilization.
- [ ] SO-101 can be teleoperated through its ROS 2 joint and gripper interfaces.
- [ ] TurtleBot4 can navigate from Point A to Point B through the ROS 2/Nav2 path without collision in the deterministic test world.
- [ ] SO-101 IK meets the documented position and orientation tolerances on reachable targets and rejects invalid targets safely.
- [ ] Domain Randomization can be enabled or disabled through `configs/sim_config.yaml` without changing ROS 2 topic contracts.
- [x] The bridge and simulator can run without Phase 2+ dependencies such as LeRobot, VLM, or VLA packages.

---

## 🎯 Phase 2: Vision-Based Motor Skills (Imitation Learning via LeRobot)

### Objective
Transition from rigid, purely mathematical calculations (Phase 1 IK/Nav2) to data-driven, vision-based control. Utilize Hugging Face's **LeRobot** ecosystem to train local neural network policies that map raw camera observations directly to low-level motor actions.

### Key Deliverables & Directory Layout
- [ ] `scripts/collect_demos.py`: Automated trajectory recorder producing a LeRobot-shaped `.npz` and metadata layout for the SO-101. **Partial:** it does not yet export the standard `LeRobotDataset` format or support teleoperation/base odometry.
- [ ] `docker/Dockerfile.lerobot`: Headless containerized environment for policy training and dependency isolation.
- [ ] `scripts/train_policy.py`: Local GPU or cloud training pipeline for ACT (Action Chunking with Transformers) or Diffusion Policy models. **Partial:** the ACT-specific `scripts/train_act.py` exists; the planned unified entry point does not.
- [ ] `scripts/eval_policy.py`: Closed-loop evaluation runner executing policies over the direct MuJoCo interface. **Partial:** scripted, replay, and LeRobot policies are supported; ROS 2 evaluation is not integrated.

### Interface Contract
* **Planned input observation:** `camera_wrist_rgb` ($224 \times 224$), `camera_top_rgb` ($224 \times 224$), `joint_states`
* **Planned output action:** Predicted joint position target sequences / base action vectors ($N=100$ chunking horizon).
* **Current prototype:** SO-101 `front` and `wrist` RGB frames, six-value joint state, and configurable ACT chunks (default `30`) in `src/physai/policy/act_dataset.py`.

### Definition of Done (DoD)
1. Harvest 50–100 successful task episodes per manipulation skill exported cleanly to `LeRobotDataset`.
2. Policy training finishes with converging loss curves inside the containerized environment.
3. Closed-loop evaluation in `eval_policy.py` achieves **> 80% task success rate** over 20 randomized trials in MuJoCo, bypassing analytical solvers.

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
