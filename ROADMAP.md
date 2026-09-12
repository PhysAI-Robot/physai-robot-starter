# 🗺️ Project Roadmap: physai-robot-starter

`physai-robot-starter` is an open-source starter kit for Embodied AI & Robotics. It bridges classical ROS 2 control stacks with visual control, learning-based motor skills, high-level vision-language planning, and end-to-end vision-language-action policies in MuJoCo.

# Project Roadmap & Strategic Vision

## Strategic Long-Term Vision
`physai-robot-starter` is designed as a **Fully Customizable, Code-First Control Framework for Embodied AI**. The core infrastructure provides a modular development baseline where engineers can plug, train, and swap any orchestration tier—ranging from primitive deterministic scripts to multi-agent foundation intelligence—directly through standard configuration entrypoints without altering the underlying communication abstractions.

The framework is structured to scale across two core vectors of customization:

### 1. Vertical Control Spectrum (Plug-and-Play Orchestration)
The execution engine supports an open, interchangeable stack of control methodologies. A developer can hot-swap or cascade these layers depending on their hardware and compute constraints:
* **The Scripted Baseline:** Hardcoded, model-free state machines for ultra-fast, predictable trajectory and path generation.
* **Classical Visual Servoing:** Closed-loop execution combining real-time camera vectors with analytical kinematic controllers tailored to the agent's specific capability profile (e.g., coordinate mapping or velocity vectoring).
* **Data-Driven Skill Learning:** An all-in-one local pipeline to log expert demonstrations into standardized robotics dataset formats, enabling native training of localized policy networks directly inside the environment.
* **Unified End-to-End VLA:** Native abstraction bindings to serve large-scale Vision-Language-Action models directly controlling physical action spaces when maximum semantic generalization is required.

### 2. Horizontal Agent Heterogeneity & Agnostic Task Routing
The simulation workspace explicitly decouples scene orchestration from physical execution to support multi-robot scaling and flexible task delegation.
* **Zero-Setup Robot Agnosticism:** To prevent onboarding complexity, all rigid physical profiles, mass properties, and meshes are entirely pre-configured and sealed within their respective **MJCF (.xml)** and **STL assets**. Users treat these agents as complete, turn-key entities. The framework interacts with them purely through unified abstract API ports—hiding hardware-specific friction from the user.
* **Centralized Semantic Orchestration:** The environment can host diverse agents simultaneously. A centralized task-routing arbiter assesses human instructions alongside environmental contexts and dynamically maps sub-tasks to the best-suited agent based on its registered capability profile (e.g., mobile bases for transit, manipulators for sorting). The arbiter can be configured as a simple deterministic decision tree, a classical behavioral tree, or scaled up to a multimodal Vision-Language Model (VLM) depending on the project's scale.


```

[Phase 1: Classical Foundation & ROS 2] ➔ [Phase 2: Learning-Based Motor Skills] ➔ [Phase 3: VLM Orchestration] ➔ [Phase 4: End-to-End VLA]

```

## Current Status

**Current phase: Phase 1 complete; Phase 1 to Phase 2 training bridge is next.**

Phase 1 is complete for the SO-101 and TurtleBot4 scope. The next work is to
make the stable contracts ready for visual-servoing and learning workflows.

Phase 2 and later are future direction only. They should consume the stable
contracts produced by Phase 1, not drive changes to those contracts ad hoc.

The roadmap tracks implementation evidence. `[x]` means the deliverable exists
and has focused coverage; `[ ]` means it is planned, missing, or partial.

---

## 🏗️ Phase 1: Classical Robotics Foundation & ROS 2 Contract (Complete)

Phase 1 is complete for the **SO-101 + TurtleBot4** scope. The main outcomes
are:

- [x] Stable capability-aware `Observation -> Action` contracts, robot registry,
  unit/frame validation, deterministic resets, and seeded regression coverage.
- [x] Reliable MuJoCo baselines for SO-101 manipulation and TurtleBot4 mobile
  control, including reproducible scripted SO-101 sorting at `20/20` success.
- [x] SO-101 ROS 2 bridge with joint, gripper, camera, TF, teleoperation, and
  real `rclpy` acceptance coverage.
- [x] TurtleBot4 ROS 2/Nav2 path with `/cmd_vel`, odometry, TF, LaserScan,
  Collision Monitor, obstacle validation, and structured navigation reports.
- [x] SO-101 FK, Jacobian, numerical IK, Cartesian targeting, joint-limit, and
  collision/contact safety validation, including explicit unreachable-target
  rejection.
- [x] Seeded domain randomization for physics, visuals, cameras, and clutter,
  with deterministic baseline preservation and evaluation metadata.
- [x] Shared acceptance coverage confirms deterministic control, ROS 2 message
  contracts, navigation without collision, IK safety, and randomization bounds.
- [x] Phase 1 runs without Phase 2+ dependencies such as LeRobot, VLM, or VLA;
  the architecture remains open for future Franka and mobile-manipulator
  adapters.

---

## Phase 1 to Phase 2 Bridge: Training Readiness

Add the smallest training boundary on top of the completed Phase 1 contracts.
This work must adapt the existing `Observation -> Action` interface for
learning tools without moving training logic into robot environments or ROS 2
adapters.

### Bridging Deliverables
- [x] `src/physai/contracts.py`: Define canonical `ObservationSpec` and `ActionSpec` schemas covering names, shapes, dtypes, units, ranges, camera layout, and normalization metadata.
- [x] `src/physai/data/gym_env.py`: Add a Gymnasium-compatible environment adapter for direct MuJoCo task training with seeded `reset()`, `step()`, spaces, `rgb_array` rendering, structured episode information, and the existing safety gate.
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
