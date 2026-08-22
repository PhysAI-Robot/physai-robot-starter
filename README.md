# PhysAI Robot Starter

Simulation-first starter project for an SO-101 robot arm. It lets you test a
pick-and-place workflow in MuJoCo before connecting a real robot or ROS2.

## What this project does

- Simulates an SO-101 arm, table, cube, and target pad.
- Runs a scripted pick-and-place policy without an API key.
- Collects demonstrations as `.npz` files.
- Replays demonstrations to check the data and action format.
- Uses SmolVLM to turn camera images and a task instruction into robot
  sub-goals.
- Keeps the interfaces ready for a future VLA policy and ROS2 integration.

## Requirements

- Python 3.10 or newer
- Linux or macOS
- A working virtual environment named `.venv`

The simulation runs on CPU. SmolVLM downloads its model on first use and may
need additional memory. A GPU is helpful but not required.

## Getting started

Follow these steps in order the first time. You do **not** need Claude or
SmolVLM to get the simulator running.

### 1. Set up the environment

From the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/fetch_assets.py
```

### 2. Run the first simulation

This runs one headless episode using the scripted policy. It is the quickest
way to check that the installation works:

```bash
python scripts/run_sim.py
```

The video is written to `outputs/`. The current built-in robot is `so101`:

```bash
python scripts/run_sim.py --robot so101
```

### 3. Open the simulator GUI (optional)

After the headless run works, open the MuJoCo viewer:

```bash
python scripts/run_sim.py --viewer
```

Close the MuJoCo window to stop it.

### 4. Run the tests (optional)

Install the development tools and run the test suite:

```bash
python -m pip install -e ".[dev]"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` avoids an unrelated ROS2 pytest plugin when
ROS2 is installed system-wide. It is not needed in an environment without
that plugin.

## Run in Docker

```bash
python docker/container.py build    # build the image
python docker/container.py start    # run the container in the background
python docker/container.py shell    # open a shell inside the running container
python docker/container.py stop     # stop and remove the container
```

Flags:

```bash
--gpu         # force the CUDA image and pass the GPU through
--cpu         # force the CPU-only image, needed on a MacBook
--no-cache    # rebuild without the layer cache
```

Without `--gpu` or `--cpu`, the GPU is used when Docker has the NVIDIA runtime
and the CPU image is used otherwise.

## Choose your next workflow

The commands below are independent options. Pick the one that matches what
you want to try; you do not need to run all of them.

### Inspect the scripted policy baseline

```bash
python scripts/eval_policy.py --policy scripted --episodes 20
```

This is the privileged SO-101 expert path. It is useful for inspecting rollout
metrics; the planner-to-`PlanRunner` path is the currently validated
pick-and-place execution baseline.

### Collect demonstrations

```bash
python scripts/collect_demos.py --episodes 50 --out data/pickplace_v1
```

### Replay demonstrations

```bash
python scripts/eval_policy.py \
  --policy replay \
  --dataset data/pickplace_v1
```

### Try SmolVLM planning

This is the first workflow that uses a model. SmolVLM reads the simulated
camera images and instruction, then proposes robot sub-goals. The first run downloads
`HuggingFaceTB/SmolVLM-500M-Instruct`.

Install its optional dependencies first:

```bash
python -m pip install -e ".[smolvlm]"
```

```bash
python scripts/plan_task.py --planner smolvlm --dry-run
```

Use a different checkpoint or device:

```bash
python scripts/plan_task.py \
  --planner smolvlm \
  --model HuggingFaceTB/SmolVLM-500M-Instruct \
  --device cpu \
  --dry-run
```

Use `--dry-run` first to inspect the generated sub-goals. Remove `--dry-run`
to execute the plan in MuJoCo.

### Check planner plumbing without a model

This needs no model download and is useful for checking the planner-to-sim
plumbing:

```bash
python scripts/plan_task.py --planner scripted --dry-run
```

### Other tools

```bash
python scripts/workspace_map.py
python scripts/teleop_keyboard.py
python scripts/show_ros2_contract.py
```

## Claude planner (optional)

Claude is still available as an alternative cloud planner:

```bash
export ANTHROPIC_API_KEY="your-key"
python scripts/plan_task.py --planner claude --dry-run
```

SmolVLM and Claude are planners. They produce sub-goals; they are not the
low-level motor policy. See [docs/architecture.md](docs/architecture.md) for
the model roles, data format, ROS2 contract, and experiment notes.

## Adding a new approach

Keep robot-specific code behind a robot factory and model-specific code behind
a planner or policy adapter. The built-in SO-101 is registered in
`src/physai/robots/registry.py`; planner backends are registered in
`src/physai/planner/registry.py`.

For a new robot, implement the common `Observation`/`Action` environment
surface, add a `RobotSpec`, and register its factory. For a new task, implement
the `Task` interface and register it independently of the robot. For a new VLM,
implement
`Planner.plan()`, register the factory, and reuse the existing `Plan` contract.
For a new action model, implement `Policy.act()` or extend the generic
`LeRobotPolicy` adapter. The simulation scripts can then select the registered
name with `--robot` or `--planner`; runtime code composes a registered task
without changing robot code.

The detailed boundaries and migration guidance are in
[docs/architecture.md](docs/architecture.md).

## Project layout

```text
src/physai/     robot contracts, simulation, planners, policies, and control
scripts/        commands for simulation, demos, evaluation, and inspection
configs/        task and simulation configuration
tests/          unit and simulation tests
docs/           architecture and research notes
assets/         downloaded robot files (ignored by Git)
data/           recorded demonstrations (ignored by Git)
outputs/        videos and generated plans (ignored by Git)
```

## Troubleshooting

If Python cannot find the project package, activate `.venv` and run commands
from the project root. If the simulator reports missing robot files, run:

```bash
python scripts/fetch_assets.py
```

If you only want to test the simulator, use `--planner scripted`; it does not
need SmolVLM, Claude, or an API key.
