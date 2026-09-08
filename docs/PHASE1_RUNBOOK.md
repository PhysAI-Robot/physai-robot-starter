# Phase 1 Runbooks

The architecture is robot-agnostic, but the operational workflow is easier to
follow when each robot has its own runbook.

Robot-owned task configurations follow `configs/tasks/<robot>/<task>.yaml`,
and robot-owned Nav2 profiles follow `configs/nav2/<robot>/`.

- [SO-101 Runbook](SO101_RUNBOOK.md): viewer, pick-and-place, ROS2, data
  collection, imitation learning, and planner/VLM/VLA continuation.
- [TurtleBot4 Runbook](TURTLEBOT4_RUNBOOK.md): viewer, velocity control, ROS2,
  RPP navigation, dummy-map Nav2, and obstacle-navigation continuation.

## Shared Setup

Run from the repository root:

```bash
uv sync --locked --extra dev
uv run python --version
```

Fetch assets for the robot you are using:

```bash
uv run python scripts/fetch_assets.py --robot so101
uv run python scripts/fetch_assets.py --robot turtlebot4
```

Run the complete regression suite:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/ -q
```

The standard environment skips tests that require real ROS2 messages. The
ROS2 Jazzy acceptance environment runs both robot-specific real-node tests.

For optional WSL2 viewer performance settings, see [README.md](../README.md).
Native Ubuntu users can skip that section.

## Shared Change Workflow

1. Run the selected robot's baseline with a fixed seed.
2. Change one owning module or parameter group.
3. Run the closest focused test.
4. Repeat the robot's direct simulation or policy evaluation.
5. Run the full suite.
6. Check the worktree:

```bash
git status --short
git diff --check
git diff --stat
```

Do not commit `.venv`, model snapshots, demonstrations, videos, generated
plans, credentials, or local evaluation output.