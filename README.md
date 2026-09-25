# PhysAI Robot Starter

Open-source starter kit for embodied AI and robotics. It connects classical
robot control, MuJoCo simulation, ROS2 interfaces, and later data-driven
policies through stable robot, task, observation, and action contracts.

**Phase 1, the Classical Foundation and ROS2 Contract, is complete** for the
supported SO-101 arm and TurtleBot4 differential-drive base in MuJoCo. The
Phase 1-to-Phase 2 training bridge is complete. Phase 2A, the classical vision
baseline, is in progress: the camera-only `visual_servo` policy has landed, but
2A is not done until it is evaluated across the T0-T4 task ladder and the
difficulty sweep. See [Roadmap](ROADMAP.md) for the definition of done.

The shortest way to inspect the completed foundation is model-free: run the
scripted SO-101 pick-and-place baseline, inspect the contracts, then validate
the ROS2 bridge and TurtleBot4 navigation acceptance path.

<p align="center">
  <img src="docs/media/so101_pick_place.gif" width="420"
       alt="SO-101 arm picking up a red cube and placing it on a green pad in MuJoCo">
</p>

<p align="center">
  <em>The Phase 1 baseline: <code>python scripts/run_sim.py</code>, one scripted
  pick-and-place episode, no model or API key involved.</em>
</p>

Every clip and screenshot below is a real simulator rollout at a fixed seed.
Regenerate them all with `python scripts/render_docs_media.py`.

## Requirements

- Ubuntu 24.04 LTS
- Python 3.12
- ROS2 Jazzy for ROS2 integration and hardware workflows
- A virtual environment

Ubuntu 24.04 and Python 3.12 are the supported baseline because ROS2 Jazzy
targets that platform. The direct MuJoCo simulator can run without a ROS2
installation, but the Docker workflow includes ROS2 Jazzy for integration
testing. VLM and VLA workflows need more memory; CUDA is optional.

## Install

Install `uv` using the instructions for your platform, then create the
environment with the common web and training extras from the project root:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra web --extra training
```

The base install contains MuJoCo, NumPy, image/video support, and YAML
configuration. The command above also installs the browser viewer and
Gymnasium training bridge. It does not install ROS2, VLM, or VLA dependencies.
Use the optional extras below when those later-phase features are needed.

Install development tools and run the test suite with:

```bash
uv sync --extra dev --extra web --extra training
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/ -q
```

## Quick start

Fetch the SO-101 description and run one scripted pick-and-place episode:

```bash
uv run python scripts/fetch_assets.py --robot so101
uv run python scripts/run_sim.py
```

The command runs headlessly by default and writes evaluation output to
`outputs/`. Add `--video` when you want a recorded episode.

For an interactive session, start the SO-101 host with `--serve` and open the
browser-based Three.js viewer — this is the actively developed client, with
configurable camera panels and per-policy debug overlays. Without an explicit
`--policy`, serve mode holds the current pose and does not run pick-and-place
automatically. Add `--policy scripted` when you want the scripted task to
drive the robot:

```bash
MUJOCO_GL=egl uv run python scripts/run_sim.py --robot so101 --serve
```

Add `--record-dir data/web_session` to record episodes from the browser with
success/fail tags (see
[docs/WEB_VIEWER_RUNBOOK.md](docs/WEB_VIEWER_RUNBOOK.md#recording-episodes)).

In another terminal, open the browser client:

```bash
MUJOCO_GL=egl uv run python scripts/run_web.py --connect http://127.0.0.1:8000
```

MuJoCo's own desktop viewer is also available as a fallback when a browser
isn't convenient; its scope stays at simulation render and basic status (see
[ADR 4](docs/adr/0004-tk-viewer-frozen.md)). Add `--viewer` in place of
`--serve` above, or alongside it to attach both clients to the same host:

```bash
uv run python scripts/run_sim.py --viewer
```

The web layer remains robot-agnostic and discovers action modes and cameras
from the host's `RobotSpec`. See the runbook for the host command, keyboard
mapping, and WebSocket/API contract.

To run multiple heterogeneous robots in one shared MuJoCo scene at the default
30 Hz control rate, use the world manifest example:

```bash
MUJOCO_GL=egl uv run python scripts/run_sim.py --world configs/worlds/heterogeneous.yaml --serve
```

This uses one model, physics data object, and simulation clock. The shared
world starts idle; the browser selector chooses which instance receives jog
commands while the whole scene remains visible. Camera frames are rendered in
a worker so camera capture does not block the physics loop. Add `--viewer` if
you also want the desktop window. See the [Web Viewer runbook](docs/WEB_VIEWER_RUNBOOK.md)
for the shared-world contract.

### Optional WSL2 viewer performance

This section is only for users running the viewer inside WSL2. Native Ubuntu
users can skip it. On WSL2, MuJoCo can fall back to the CPU software renderer
(`llvmpipe`), which makes the interactive viewer look choppy even when
`nvidia-smi` can see the NVIDIA GPU. If that happens, enable the WSLg D3D12
renderer for the shell before opening the viewer:

```bash
export GALLIUM_DRIVER=d3d12
uv run python scripts/run_sim.py --viewer
```

To apply this automatically to future Bash sessions, add the setting once:

```bash
printf '\nexport GALLIUM_DRIVER=d3d12\n' >> ~/.bashrc
source ~/.bashrc
```

Verify that OpenGL is accelerated and reports the NVIDIA GPU:

```bash
glxinfo -B | grep -Ei 'vendor|renderer|accelerated'
```

The renderer should mention `D3D12` and the NVIDIA GPU, not `llvmpipe` or
`Accelerated: no`. WSL2 GPU support requires a current NVIDIA driver on the
Windows host and WSLg; do not install the Linux NVIDIA display driver inside
WSL with `sudo apt install nvidia-driver`.

Run the same task from the checked-in YAML configuration:

```bash
uv run python scripts/run_sim.py --config configs/tasks/so101/pick_place.yaml
```

Use `--seed`, `--max-steps`, and `--camera-size` to override configuration.
The shared simulation seed and domain-randomization switch come from
`configs/sim_config.yaml`, selected by `--sim-config` and defaulting to that
file. Domain randomization can be enabled through that configuration; keep it
disabled for the deterministic baseline and use
`scripts/eval_randomization.py` to compare deterministic and randomized runs.
For image-conditioned policies, keep `--camera-size` square and match the
training resolution, such as `128` or `224`.

## Current Phase 1 scope

Two embodiments are supported, and they do not share an action space: the arm
takes joint positions, the base takes a twist.

| SO-101 | TurtleBot4 |
| --- | --- |
| <img src="docs/media/so101_pick_place.gif" width="330" alt="SO-101 arm performing scripted pick-and-place"> | <img src="docs/media/turtlebot4_drive.gif" width="330" alt="TurtleBot4 driving an arc across a checkered floor"> |
| Scripted pick-and-place, joint-position control | Constant forward and yaw twist, differential drive |

| Robot | Phase 1 result | Next extension |
| --- | --- | --- |
| SO-101 | Deterministic pick-and-place, ROS2 joint/gripper/camera/TF bridge, and IK safety validation | Visual servoing and learned manipulation |
| TurtleBot4 | Deterministic navigation, ROS2/Nav2 acceptance path, obstacle detection, and collision telemetry | Visual goal tracking and future hardware integration |

The TurtleBot4 path includes a deterministic ROS2 interface, open-space Nav2
smoke testing, obstacle-aware navigation, LaserScan validation, and MuJoCo
collision telemetry. The SO-101 path includes the ROS2 joint, gripper, camera,
and TF bridge. Controlled domain randomization is available behind the
configuration toggle and remains disabled by default.
Direct MuJoCo remains the fast local path and does not replace ROS2
integration validation.

The planned progression is:

```text
Phase 1  Classical foundation + ROS2 contract
        -> Phase 2  Benchmark, baselines, and learning
          2.0  Task ladder and capability report
          2A   Classical vision baseline (camera-only)
          2B   Imitation learning with ACT
          2C   Backend comparison study
          2D   Report and release (v0.2)
    -> Phase 3 and 4  Not in focus yet
```

Phase 2 and later are future direction. Their current scripts and adapters are
experimental seams around the Phase 1 contracts, not completion claims for
those phases. See [Roadmap](ROADMAP.md) for the scope and definition of done
for each phase.

The Phase 1-to-Phase 2 bridge now includes canonical `ObservationSpec` and
`ActionSpec` schemas plus a Gymnasium adapter. The standard setup above already
installs its training dependency; for a base-only environment, add it with:

```bash
uv sync --extra training
```

The bridge is ready for training integration; the remaining Phase 2 work is
tracked in the [Roadmap](ROADMAP.md).

## Phase 1 workflows

### What a policy observes

The SO-101 publishes two camera views per step alongside joint state. Both are
recorded into every demonstration and are the only inputs an image-conditioned
policy receives; the scripted expert ignores them and reads cube pose straight
from the simulator instead.

| `front` | `wrist` |
| --- | --- |
| <img src="docs/media/so101_camera_front.png" width="300" alt="Front camera view of the arm, red cube, and green target pad"> | <img src="docs/media/so101_camera_wrist.png" width="300" alt="Wrist camera view looking down at the jaws closing on the red cube"> |
| Fixed world view: arm, cube, and target pad | Gripper-mounted, looking down the approach axis |

Both frames come from `observation.images` on the same timestep, captured here
as the jaws close on the cube.

### Evaluate and replay

Evaluate the scripted expert over multiple seeds:

```bash
uv run python scripts/eval_policy.py --policy scripted --episodes 20
```

Record successful demonstrations, then replay their actions through the
simulator:

```bash
uv run python scripts/collect_demos.py --episodes 50 --out data/pickplace_v1
uv run python scripts/eval_policy.py --policy replay --dataset data/pickplace_v1
```

The sorting variant records three cubes and chooses a target color per episode:

```bash
uv run python scripts/collect_demos.py --sorting --episodes 50 --out data/sorting_v1
```

<p align="center">
  <img src="docs/media/sorting/sorting_scripted.gif" width="360"
       alt="SO-101 arm selecting the blue cube from three colored cubes and placing it on the pad">
</p>

Measured over held-out seeds, the scripted expert now reaches 100% on the
single-cube check (300 seeds) and 98% on this sorting variant (900 seeds).
The single-cube number matches the camera-only `visual_servo` baseline; see
[research/scripted_experts/README.md](research/scripted_experts/README.md)
for the root causes behind the sorting gap and their fixes: a missing
wrist-orientation constraint, an under-squeezed grip, and the wide-open jaws
nudging a neighboring cube during approach and staling the expert's locked
aim point.

Failed demonstrations are discarded by default. Add `--keep-failures` when
you are analyzing failure cases.

### Development inspection

```bash
uv run python scripts/workspace_map.py
uv run python scripts/show_ros2_contract.py
uv run python scripts/teleop_keyboard.py
uv run python scripts/export_scene.py --out outputs/scene_pick_place.xml
uv run python scripts/render_docs_media.py --only so101
```

`teleop_keyboard.py` opens the MuJoCo viewer and exercises Cartesian jogging.
`export_scene.py` writes a composed MJCF file; its relative mesh paths resolve
from the SO-101 asset directory. `render_docs_media.py` re-renders the clips
and screenshots this README embeds, which is also a quick way to eyeball
whether a scene or camera change looks right.

## Later-phase experiments

The repository contains early data and model workflows so they can be tested
against the shared contracts. They belong to the roadmap's later phases and
are not required for the completed Phase 1 baseline or its training bridge.

### Phase 2B: imitation learning with ACT

Collect demonstrations and fine-tune an ACT policy after installing the VLA
extra:

```bash
uv sync --extra vla
uv run python research/imitation_learning/train_act.py --dataset data/pickplace_v1 --steps 4000
uv run python scripts/eval_policy.py --policy lerobot --checkpoint outputs/act_ckpt --camera-size 128
```

The training script stores the checkpoint and metadata under
`outputs/act_ckpt` by default. Use the same square image size during training
and evaluation to avoid an image distribution mismatch.

## Python API

`physai.runtime.create_runtime` composes a registered robot, task, policy, and
safety boundary:

```python
from physai.runtime import create_runtime

runtime = create_runtime("so101", task_name="pick_place")
observation = runtime.reset(seed=0)
try:
    # Pass actions from a policy or resolver here.
    pass
finally:
    runtime.close()
```

For the multi-object sorting task, select the matching scene explicitly:

```python
runtime = create_runtime(
    "so101",
    scene_name="sorting_minimal",
    task_name="sorting",
)
```

## Optional models

Install the extra for the workflow you intend to use:

```bash
# ACT / LeRobot policy support (Phase 2B)
uv sync --extra vla
```

Checkpoints are written by `research/imitation_learning/train_act.py` into the ignored local
`outputs/` directory and loaded from an explicit path.

Copy `.env.example` to `.env` when using gated or private Hugging Face models.
Do not commit `.env`, credentials, checkpoints, or downloaded assets.

## ROS2 Jazzy

ROS2 integration targets Jazzy on Ubuntu 24.04. The repository's Docker image
installs `ros-jazzy-ros-base`, `ros-jazzy-rviz2`, `ros-jazzy-ros2-control`, and
`ros-jazzy-ros2-controllers`, then sources `/opt/ros/jazzy/setup.bash` when the
container starts.

For a host installation, follow the official
[ROS2 Jazzy Ubuntu installation guide](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)
before using ROS2 tools. Inspect the message-shaped contract without ROS2 by
running:

```bash
uv run python scripts/show_ros2_contract.py
```

The first synchronous MuJoCo bridge core is available as
`physai.bridge.MuJoCoROSBridge`. It uses an injected transport, publishes
joint states and rendered camera images, accepts joint trajectory and gripper
commands, and applies the shared safety gate. The real `rclpy` nodes also
publish TF and CameraInfo for SO-101 and odometry, scans, and TF for
TurtleBot4; the acceptance paths are documented in the robot runbooks.

## Docker

The helper chooses the GPU image by default. Select CPU mode while building on
machines without the NVIDIA runtime:

```bash
python docker/container.py build --cpu
python docker/container.py start
python docker/container.py shell
python docker/container.py stop
```

Use `build --gpu` for the CUDA image and `build --no-cache` to rebuild without
cached layers. Rebuild to switch between GPU and CPU modes.

## Generated files and licenses

The following local directories are ignored by Git and Docker:

- `assets/`: downloaded robot descriptions and meshes
- `data/`: recorded demonstrations
- `outputs/`: videos, plans, checkpoints, and evaluation artifacts

The PhysAI Robot Starter source code is licensed under the Apache License 2.0.
Downloaded robot assets, model checkpoints, and Python dependencies retain
their original licenses. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
before redistributing downloaded artifacts. The TurtleBot4 model attribution
is also documented there.

## Documentation

- [Architecture](docs/ARCHITECTURE.md): runtime composition, module ownership,
  contracts, and extension boundaries.
- Runbooks: [SO-101](docs/SO101_RUNBOOK.md),
  [TurtleBot4](docs/TURTLEBOT4_RUNBOOK.md), and the
  [web viewer](docs/WEB_VIEWER_RUNBOOK.md).
- [Migration record](docs/MIGRATION.md): the core-architecture-freeze
  restructuring, old path -> new path.
- [Architecture decisions](docs/adr/): the decisions behind the frozen design.
- [Contributing](CONTRIBUTING.md): contribution workflow and commit format.
- [Third-party notices](THIRD_PARTY_NOTICES.md): asset and model sources,
  attribution, and license status.
- [Agent guide](AGENTS.md): rules for coding agents working in this repository.
- [Roadmap](ROADMAP.md): planned work and migration direction.

## Troubleshooting

Run commands from the project root with `uv run`. If robot files are
missing, fetch them again:

```bash
uv run python scripts/fetch_assets.py
```

For a simulator-only check, use `--policy constant` or the default scripted
SO-101 policy. Viewer and serve modes are different: they stay idle unless a
policy is explicitly supplied, so they can be controlled manually from the
browser. None of these workflows requires a model download or an API key. If video encoding is unavailable, the simulator falls back to a
GIF; installing `imageio-ffmpeg` enables MP4 output.
