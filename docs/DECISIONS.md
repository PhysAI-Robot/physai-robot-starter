# Design decisions

The few decisions that shape the frozen design and are not obvious from the code. Each
section is a rule plus its reason; the module layout and contracts are in
[ARCHITECTURE.md](ARCHITECTURE.md). Add a new decision as the next letter, and change an
existing one in place when it no longer holds (git history keeps the old text).

## A. Research code lives outside the core package

`src/physai` holds the frozen infrastructure (contracts, ports, registries) and
technique-agnostic baselines (`ScriptedPlanner`, `PlanRunner`, `SortingTask`, `ReplayPolicy`,
the Gymnasium adapter, TurtleBot4 navigation). Approach-specific code (scripted expert,
visual servo, ACT, VLA adapter, VLM planner) lives in `research/<topic>/` and registers itself
with a core registry (`policy.registry`, `planner.registry`, `robots.registry`) on import. A
new technique needs a package and a registration call, never a core change.

Core never imports `research/`, and the other dependency rules (`sim` must not import ROS2,
`policy`/`tasks`/`planner` must not import MuJoCo, the engines must not import each other) are
declared in `pyproject.toml`'s `[tool.importlinter]` and checked by `uv run lint-imports` and
`tests/core/boundaries/test_import_boundaries.py`, so `pytest tests/ -q` fails on a violation.
import-linter is a declarative tool, not a hand-written import walker.

## B. The manifest is the only run description

A `SessionManifest` (`physai/config/manifest.py`, files in `configs/manifests/`) describes a
run: robots (`{id, robot, pose, task?, policy?}`), scene or `world`, `backend`
(`direct | ros2_sim | ros2_real`, the last accepted but not implemented), `simulator`, `viewer`
and a `simulation` block. A single-robot run is a manifest with one entry. The field list and
validation rules are in the module docstring.

- `physai.runtime.create_session()` builds a run from a manifest; `robot_env_config` gives the
  ROS2 node the same robot fields and scene. There is no second loader: `--robot`, `--config`,
  `--world`, `configs/tasks/` and `configs/sim_config.yaml` are gone.
- One source per default: the `simulation` block owns seed and randomization (a robot `config`
  repeating them is rejected), the task owns success defaults, the robot registry owns a
  robot's default task. `max_steps` is the manifest's (`--max-steps` overrides it).
- `run_sim.py` (run and look), `eval_policy.py` (measure) and `collect_demos.py` share the
  session and `physai.runtime.run_episode`, which ends an episode on termination, truncation,
  `policy.done` or a `SafetyViolation` and raises `RenderGlitch` when Isaac stops drawing the
  robot. Video and recording are `EpisodeObserver`s. Difficulty flags become manifest overrides
  through `config.compat.with_overrides`, which re-validates the result.
- A host-driven (viewer or serve) session steps the bare robot and composes no task. The host's
  camera thread renders only for robots that accept `camera_stride=0`; TurtleBot4 renders
  inline because a second GL context on another thread fails on Windows.
- One flag name per concept (`--out DIR`, `--json FILE`, `--dataset DIR`, `--max-steps`), built
  by `scripts/_cli.new_parser`; `tests/core/unit/test_cli_conventions.py` rejects retired names
  and flags without help. Old names are removed, not aliased.

## C. Sim-neutral data

Everything both engines read is free of any simulator SDK (import-linter enforces it):

- `physai.robots.description.RobotDescription`: frames, cameras, contact pads, actuators and
  per-simulator `sim_overrides`, loaded from one `description.yaml` per robot (metres, `wxyz`
  quaternions, radians). Each engine has one builder. Solver tuning with no cross-engine
  equivalent (MuJoCo `solref`/`solimp`) lives under `sim_overrides` and is re-tuned per engine.
  The SO-101 pad quaternions are baked forward-kinematics output; a test re-derives them to 1e-6.
- `ImageFrame` carries optional `intrinsics` (`CameraIntrinsics`) and `extrinsics` (ROS optical
  convention, x right, y down, z forward), so a vision policy reads calibration off the
  observation and works on any backend that fills them. `contracts.py` is frozen, which is why
  this needed a decision.
- `physai.sim.workspace`: `WorkspaceConfig` (table, target, front camera, `cubes()`), `CubeSpec`
  and the shared constants (`TABLE_FRICTION`, `CUBE_FRICTION`, `TARGET_RGBA`,
  `FRONT_CAMERA_FOVY_DEG`), so a value both engines must agree on is written once and a test
  checks the compiled MuJoCo scene against it.

## D. Isaac Sim is an optional peer engine

Goal: evaluate a MuJoCo-tuned policy on a second engine to measure the sim-to-sim gap.

- **Layout.** `physai/sim/mujoco/` and `physai/sim/isaac/` are peers; `sim/__init__.py` is a
  docstring so importing `physai.sim` never needs an engine SDK. Only `physai.sim.isaac` and
  `robots.so101.isaac_env.SO101IsaacEnv` may import `isaacsim`/`omni`/`pxr`, always
  function-locally (they are not import-safe before a `SimulationApp` exists). Adapters match the
  manifest `backend` one to one: `DirectAdapter`, `ROS2SimAdapter`, `ROS2HardwareAdapter`.
- **Selection.** `RobotDescriptor.simulators` (default `("mujoco",)`; so101 also `isaac`) and the
  manifest's top-level `simulator`, validated at load: every robot must support it, and Isaac
  rejects a `world` block, `backend: ros2_sim` and any `viewer.mode` but `none`. `--sim isaac`
  overrides any manifest, so `so101_single_cube_fixed_place.yaml` is the one session both engines
  run.
- **Same scene.** `IsaacEnvConfig.scene` takes the scene MuJoCo builds from; the per-seed cube
  and target positions come from `layout.draw_xy`, pinned by `golden_layouts.json`. Limits:
  single-cube scenes, no randomization beyond the cube and target spawn, a lighting scale and
  camera jitter, observation-only policies (`visual_servo`, `constant`, `lerobot`); `--viewer`, web recording
  and playback, `SharedWorld`, the ROS2 bridge and `ArmKinematics` stay MuJoCo-only. Isaac
  `--serve` runs the `Host` on the main thread (Isaac must stay on the thread that created
  `SimulationApp`) with uvicorn on a worker thread, and shows a display mirror refreshed from
  Isaac's joint and cube state.
- **Install.** `isaac` is a normal extra (`uv sync --extra isaac --extra training`); it needs the
  NVIDIA index and `index-strategy = "unsafe-best-match"`, which `pyproject.toml` documents.
  `lerobot` is part of `training` so one checkpoint runs on both engines. Because
  `lerobot` wants `numpy<2.3`/`packaging<26` and Isaac pins `numpy==2.3.1`/`packaging==26.0`,
  `[tool.uv] override-dependencies` forces `numpy>=2.0,<2.3` and `packaging>=24.2,<26.0`;
  `uv pip check` lists the broken pins. When a release needs `numpy` 2.3 the overrides go.
  `webtest` (`httpx2`) conflicts with `isaac` (`idna`). On Windows `torch` comes from the `cu128`
  index. `OMNI_KIT_ACCEPT_EULA=YES` is set by the person running Isaac, never by this repo.
  `lerobot`'s `draccus` installs a top-level `tests` package, hence `tests/__init__.py`.

## E. One `Host`, a frozen desktop viewer, record and replay through the data path

- **One class.** `physai.web.host.Host` serves one robot or a shared world (a single robot is a
  world with one instance). Every method is instance-keyed, the id may be omitted for one robot,
  reset and pause are world-atomic, and `web/app.py` never branches on host type.
- **`--viewer` is MuJoCo's own viewer** (`mujoco.viewer.launch_passive`) and stays limited to
  rendering and basic status; camera panels, jog, instance selection, telemetry and overlays go
  into the web viewer only. It renders a private `MjData` copy refreshed under
  `Host.physics_lock`.
- **Record and replay** reuse `EpisodeRecorder`; the web layer has no episode format. A world
  reset or a policy ending its episode discards the take. Playback exists only while paused: it
  restores recorded state and calls `mj_forward`, with no re-simulation, and refuses commands,
  resets and recording meanwhile. Episodes may carry `observation.environment_state` (full
  `qpos`, float64), which `collect_demos.py` sets; older datasets cannot be played back, and a
  different `nq` is rejected. Both are single-instance only.
