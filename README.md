# PhysAI Robot Starter

Simulation-first robot stack for testing multiple robot embodiments in MuJoCo
before connecting real hardware or ROS2. The built-in adapters are an SO-101
arm and a TurtleBot4 differential-drive base.

The default path is deliberately local and model-free: run the scripted
SO-101 pick-and-place episode first, then add a planner or action model when
the simulator is working.

## Requirements

- Python 3.10 or newer
- Linux or macOS
- A virtual environment named `.venv`

The simulator runs on CPU. VLM/VLA inference may need additional memory; a GPU
is helpful but not required.

## Quick start

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/fetch_assets.py --robot so101
python scripts/run_sim.py
```

The first run uses the scripted policy and writes a video to `outputs/`. Open
the MuJoCo viewer after the headless check succeeds:

```bash
python scripts/run_sim.py --viewer
```

Run the generic TurtleBot4 smoke test with:

```bash
python scripts/fetch_assets.py --robot turtlebot4
python scripts/run_sim.py --robot turtlebot4 --policy constant --no-video
```

## Common workflows

Install development tools before running tests:

```bash
python -m pip install -e ".[dev]"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q
```

Useful commands:

```bash
# Inspect the scripted SO-101 baseline
python scripts/eval_policy.py --policy scripted --episodes 20

# Record and replay demonstrations
python scripts/collect_demos.py --episodes 50 --out data/pickplace_v1
python scripts/eval_policy.py --policy replay --dataset data/pickplace_v1

# Check planner-to-simulator plumbing without a model
python scripts/plan_task.py --planner scripted --dry-run

# Run local SmolVLM planning
python -m pip install -e ".[smolvlm]"
python scripts/plan_task.py --planner smolvlm --dry-run

# Inspect workspace and ROS2-shaped contracts
python scripts/workspace_map.py
python scripts/show_ros2_contract.py
```

The planner commands are SO-101-specific. `scripted` needs no API key or model
download. SmolVLM reads simulated camera images and proposes sub-goals; it is
not the low-level motor policy.

## Optional models

Download model snapshots into the local, ignored `models/` directory:

```bash
python -m pip install huggingface-hub
python scripts/download_models.py --model smolvlm
python scripts/download_models.py --model smolvla
python scripts/download_models.py --model turbovla
```

Copy `.env.example` to `.env` and set `HF_TOKEN` only for gated/private
Hugging Face models or when the anonymous API limit is reached. Claude is an
optional cloud planner:

```bash
export ANTHROPIC_API_KEY="your-key"
python scripts/plan_task.py --planner claude --dry-run
```

Model loading uses explicit local paths and does not silently populate the
global Hugging Face cache. VLA checkpoints can be passed with `--checkpoint`.

## Docker

```bash
python docker/container.py build
python docker/container.py start
python docker/container.py shell
python docker/container.py stop
```

Use `--gpu`, `--cpu`, or `--no-cache` with `build` when needed. Without an
override, the helper chooses the GPU image when the NVIDIA runtime is
available and otherwise uses the CPU image.

## Local files

Generated files are intentionally kept out of the source tree's tracked code:

- `assets/`: downloaded robot descriptions and meshes
- `data/`: recorded demonstrations
- `models/`: local model snapshots
- `outputs/`: videos, plans, and evaluation artifacts

## Documentation map

- [Architecture](docs/ARCHITECTURE.md): runtime composition, module ownership,
  contracts, and extension boundaries.
- [Agent guide](AGENTS.md): rules for coding agents working in this repository.
- [Roadmap](ROADMAP.md): planned work and migration direction.

The TurtleBot4 MuJoCo model is adapted from
[`narcispr/turtlebot4_mujoco`](https://github.com/narcispr/turtlebot4_mujoco),
with the original model noted there as originating from the HTWK Leipzig
`ai-enhanced-ros` project. Review the upstream license before redistribution.

## Troubleshooting

Run commands from the project root with `.venv` activated. If robot files are
missing, fetch them again:

```bash
python scripts/fetch_assets.py
```

For a simulator-only check, use `--policy constant` or the default scripted
SO-101 policy; neither requires SmolVLM, Claude, or an API key.
