# PhysAI Robot Starter

Open-source starter kit for embodied AI and robotics. It connects classical
robot control, MuJoCo simulation, ROS2 interfaces, and later data-driven
policies through stable robot, task, observation, and action contracts.

**Status:** Phase 1 (classical foundation and ROS2 contract) and the Phase 1-to-2
training bridge are complete for the supported SO-101 arm and TurtleBot4
differential-drive base in MuJoCo. Phase 2A, the classical vision baseline, is
in progress. See the [Roadmap](ROADMAP.md) for scope and definitions of done.

<p align="center">
  <img src="docs/media/so101_pick_place.gif" width="420"
       alt="SO-101 arm picking up a red cube and placing it on a green pad in MuJoCo">
</p>

<p align="center">
  <em>The Phase 1 baseline: <code>python scripts/run_sim.py</code>, one scripted
  pick-and-place episode, no model or API key involved.</em>
</p>

Every clip and screenshot in the docs is a real simulator rollout at a fixed
seed. Regenerate them with `python scripts/render_docs_media.py`.

## Requirements

- Ubuntu 24.04 LTS, Python 3.12, and a virtual environment
- ROS2 Jazzy for ROS2 integration (the Docker image includes it)

Ubuntu 24.04 and Python 3.12 are the supported baseline because ROS2 Jazzy
targets that platform. The direct MuJoCo simulator runs without ROS2. VLM and
VLA workflows need more memory; CUDA is optional.

## Install

Install `uv` for your platform, then create the environment from the project
root:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra web --extra training
```

The base install contains MuJoCo, NumPy, image/video support, and YAML
configuration; `web` adds the browser viewer and `training` the Gymnasium
bridge. ROS2, VLM, and VLA dependencies are separate (`--extra vla` for
ACT/LeRobot). To run the test suite:

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

It runs headlessly and writes evaluation output to `outputs/`; add `--video`
to record the episode. A run is described by a session manifest
(`--manifest configs/manifests/so101_pick_place.yaml` is the checked-in task;
its `simulation` block holds the seed and the domain-randomization switch,
which stays off for the deterministic baseline), and `--seed`, `--max-steps`,
and `--camera-size` override it. Image-conditioned policies need a square
`--camera-size` matching their training resolution, such as `128` or `224`.

### Interactive viewer

Start the host with `--serve` and open the browser client (Three.js, with
configurable camera panels and per-policy debug overlays):

```bash
MUJOCO_GL=egl uv run python scripts/run_sim.py --robot so101 --serve
MUJOCO_GL=egl uv run python scripts/run_web.py --connect http://127.0.0.1:8000
```

Serve mode holds the current pose until you pass `--policy` (for example
`--policy scripted`). Add `--record-dir data/web_session` to record episodes
from the browser. `--viewer` opens MuJoCo's own desktop window instead of, or
alongside, `--serve` ([ADR 4](docs/adr/0004-tk-viewer-frozen.md)).

To run several robots in one MuJoCo scene, one model, physics data object, and
clock:

```bash
MUJOCO_GL=egl uv run python scripts/run_sim.py --manifest configs/manifests/heterogeneous_world.yaml --serve
```

The [Web Viewer runbook](docs/WEB_VIEWER_RUNBOOK.md) covers the keyboard
mapping, recording and playback, the shared-world contract, and troubleshooting
(including WSL2 and dual-GPU Windows performance).

## Phase 1 scope

Two embodiments are supported and do not share an action space: the arm takes
joint positions, the base takes a twist.

| SO-101 | TurtleBot4 |
| --- | --- |
| <img src="docs/media/so101_pick_place.gif" width="330" alt="SO-101 arm performing scripted pick-and-place"> | <img src="docs/media/turtlebot4_drive.gif" width="330" alt="TurtleBot4 driving an arc across a checkered floor"> |
| Scripted pick-and-place, joint-position control, ROS2 joint/gripper/camera/TF bridge, IK safety validation | Deterministic navigation, ROS2/Nav2 acceptance path, obstacle detection, collision telemetry |

Controlled domain randomization is available behind the configuration toggle
and is off by default. Direct MuJoCo is the fast local path and does not
replace ROS2 integration validation. The canonical `ObservationSpec` and
`ActionSpec` schemas plus a Gymnasium adapter form the training bridge; the
remaining Phase 2 work is tracked in the [Roadmap](ROADMAP.md).

### What a policy observes

The SO-101 publishes two camera views per step alongside joint state. Both are
recorded into every demonstration and are the only inputs an image-conditioned
policy receives; the scripted expert ignores them and reads cube pose from the
simulator.

| `front` | `wrist` |
| --- | --- |
| <img src="docs/media/so101_camera_front.png" width="300" alt="Front camera view of the arm, red cube, and green target pad"> | <img src="docs/media/so101_camera_wrist.png" width="300" alt="Wrist camera view looking down at the jaws closing on the red cube"> |
| Fixed world view: arm, cube, and target pad | Gripper-mounted, looking down the approach axis |

## Workflows

Evaluate the scripted expert, then record demonstrations and replay them:

```bash
uv run python scripts/eval_policy.py --policy scripted --episodes 100
uv run python scripts/collect_demos.py --episodes 50 --out data/pickplace_v1
uv run python scripts/eval_policy.py --policy replay --dataset data/pickplace_v1
```

Add `--sorting` to `eval_policy.py` and `collect_demos.py` for the three-cube
sorting variant, and `--keep-failures` to keep failed demonstrations (they are
discarded by default). Measured success rates and how to reproduce them are in
[research/scripted_experts/README.md](research/scripted_experts/README.md);
use at least 100 seeds, since 20 cannot resolve a policy's reliability.

<p align="center">
  <img src="docs/media/sorting/sorting_scripted.gif" width="360"
       alt="SO-101 arm selecting the blue cube from three colored cubes and placing it on the pad">
</p>

Inspection helpers:

```bash
uv run python scripts/workspace_map.py
uv run python scripts/show_ros2_contract.py
uv run python scripts/teleop_keyboard.py
uv run python scripts/export_scene.py --out outputs/scene_pick_place.xml
uv run python scripts/render_docs_media.py --only so101
```

`teleop_keyboard.py` opens the MuJoCo viewer and exercises Cartesian jogging;
`export_scene.py` writes a composed MJCF file whose relative mesh paths resolve
from the SO-101 asset directory.

Later-phase experiments (ACT imitation learning, the classical visual-servo
baseline) live under `research/`; each topic's README is its runbook:
[imitation learning](research/imitation_learning/README.md),
[classical control](research/classical_control/README.md),
[scripted experts](research/scripted_experts/README.md). Copy `.env.example` to
`.env` for gated Hugging Face models, and never commit `.env`, credentials,
checkpoints, or downloaded assets.

## ROS2 Jazzy

ROS2 integration targets Jazzy on Ubuntu 24.04; the Docker image installs
`ros-jazzy-ros-base`, `ros-jazzy-rviz2`, `ros-jazzy-ros2-control`, and
`ros-jazzy-ros2-controllers`. For a host install, follow the
[official guide](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html).
The message-shaped contract can be inspected without ROS2:

```bash
uv run python scripts/show_ros2_contract.py
```

`physai.bridge.MuJoCoROSBridge` is the synchronous bridge core (injected
transport, joint states and camera images out, joint trajectory and gripper
commands in, shared safety gate). The real `rclpy` nodes add TF and CameraInfo
for SO-101 and odometry, scans, and TF for TurtleBot4; the acceptance paths are
in the robot runbooks.

## Python API

`physai.runtime.create_runtime` composes a registered robot, task, policy, and
safety boundary:

```python
from physai.runtime import create_runtime

runtime = create_runtime("so101", task_name="pick_place")
observation = runtime.reset(seed=0)
try:
    ...  # pass actions from a policy or resolver here
finally:
    runtime.close()
```

For the sorting task, select its scene explicitly with
`create_runtime("so101", scene_name="sorting_minimal", task_name="sorting")`.

## Docker

```bash
python docker/container.py build --cpu    # or --gpu for the CUDA image
python docker/container.py start
python docker/container.py shell
python docker/container.py stop
```

The helper builds the GPU image by default; use `--no-cache` to rebuild without
cached layers, and rebuild to switch between GPU and CPU modes.

## Generated files and licenses

`assets/` (downloaded robot descriptions), `data/` (demonstrations), and
`outputs/` (videos, plans, checkpoints, evaluation artifacts) are ignored by
Git and Docker.

The source code is Apache License 2.0. Downloaded robot assets, model
checkpoints, and Python dependencies keep their original licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before redistributing
downloaded artifacts.

## Documentation

- [Architecture](docs/ARCHITECTURE.md): runtime composition, module ownership,
  contracts, and extension boundaries.
- Runbooks: [SO-101](docs/SO101_RUNBOOK.md),
  [TurtleBot4](docs/TURTLEBOT4_RUNBOOK.md), and the
  [web viewer](docs/WEB_VIEWER_RUNBOOK.md). Session manifests live in
  `configs/manifests/`, Nav2 profiles at `configs/nav2/<robot>/`, and maps at
  `configs/maps/<environment>/`.
- [Architecture decisions](docs/adr/): the decisions behind the frozen design.
- [Contributing](CONTRIBUTING.md): workflow and commit format.
- [Agent guide](AGENTS.md): rules for coding agents working in this repository.
- [Roadmap](ROADMAP.md): planned work and definitions of done.
- [Third-party notices](THIRD_PARTY_NOTICES.md): asset and model sources,
  attribution, and license status.

## Troubleshooting

Run commands from the project root with `uv run`. If robot files are missing,
run `uv run python scripts/fetch_assets.py` again. For a simulator-only check,
use `--policy constant` or the default scripted SO-101 policy; viewer and serve
modes stay idle unless a policy is supplied. None of these workflows needs a
model download or an API key. If no H.264 encoder is available the simulator
falls back to a GIF; installing `imageio-ffmpeg` enables MP4 output.
