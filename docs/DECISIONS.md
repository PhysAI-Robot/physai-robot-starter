# Design decisions

The few decisions that shape the frozen design and are not obvious from the code. Each
section is a rule plus its reason; the module layout and contracts are in
[ARCHITECTURE.md](ARCHITECTURE.md). Add a new decision as the next letter, and change an
existing one in place when it no longer holds (git history keeps the old text).

## A. Research code lives outside the core package

`src/physai` holds the frozen infrastructure (contracts, ports, registries) and
technique-agnostic baselines (`ScriptedPlanner`, `PlanRunner`, `SortingTask`, `ReplayPolicy`,
the Gymnasium adapter, TurtleBot4 navigation, the SO-101 `demo` policy). Approach-specific code
(scripted expert, visual servo, ACT, VLA adapter, VLM planner) lives in a plugin module, for
this project in `research/<topic>/`, and registers itself with a core registry
(`policy.registry`, `planner.registry`, `robots.registry`, scenes, tasks) on import. A new
technique needs a package and a registration call, never a core change.

Core never imports `research/`: not `src/physai`, not `scripts/`, not `tests/core/` (an AST scan
in `tests/core/boundaries/test_import_boundaries.py`, because `research/` may be absent and
import-linter cannot check a missing package). The other
dependency rules (`sim` must not import ROS2, `policy`/`tasks`/`planner` must not import MuJoCo,
the engines must not import each other) are declared in `pyproject.toml`'s
`[tool.importlinter]` and checked by `uv run lint-imports` and the same test file, so
`pytest tests/ -q` fails on a violation.

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
- `run_sim.py` (run and look) and `eval_policy.py` (measure, and record episodes with `--save-dataset`)
  share the session and `physai.runtime.run_episode`, which ends an episode on termination, truncation,
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
  the arms' `isaac_env` modules (`robots.so101`, `robots.viperx300s`) may import `isaacsim`/`omni`/`pxr`, always
  function-locally (they are not import-safe before a `SimulationApp` exists). Adapters match the
  manifest `backend` one to one: `DirectAdapter`, `ROS2SimAdapter`, `ROS2HardwareAdapter`.
- **Selection.** `RobotDescriptor.simulators` (default `("mujoco",)`; the SO-101 and ViperX 300 S also `isaac` and `newton`) and the
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
- **Install.** One extra, `isaac`. A plain `uv sync` installs MuJoCo, Newton and the training stack (torch, lerobot),
  and `uv sync --extra isaac` adds Isaac Sim (several GB, an NVIDIA GPU, an accepted EULA; no macOS wheels, a marker
  skips it there). Nothing in the default environment differs from the Isaac one: the lock holds one version of every
  package, because the default is pinned to what `isaacsim-core` 6.2.0.0 pins (`newton==1.6.1`, hence `mujoco` and
  `mujoco-warp` 3.12.0) and the resolver takes the rest from `isaacsim-kernel` (`uvicorn` 0.29.0, `fastapi`,
  `websockets`, `idna` and others, older than the newest), so adding the extra moves no version and cannot conflict.
  Isaac Sim needs the NVIDIA index and `index-strategy = "unsafe-best-match"`, which `pyproject.toml` documents.
  `lerobot` wants `numpy<2.3` and Isaac pins `numpy==2.3.1`, so the one `[tool.uv] override-dependencies` entry
  forces `numpy>=2.0,<2.3` (Isaac runs on 2.2.x; `uv pip check` lists the broken pin); when a release needs `numpy`
  2.3 it goes. Isaac Sim's livestream and web extensions are not exercised.
  **`isaacsim` is held at 6.2.0.0 on purpose.** Study 01 compares Isaac Sim results across its runs, so
  every Isaac Sim number is measured on one version, and MuJoCo, mujoco-warp and Newton are whatever it pins
  (3.12.0, 3.12.0, 1.6.1), not the 3.13 of the study's first runs. A bump means re-measuring the Isaac Sim
  rows that a comparison uses. `.github/dependabot.yml` allows only `isaacsim` and the few packages Isaac does not pin; a Dependabot
  bump of `isaacsim` cannot resolve alone (`newton` is pinned beside it), so it is a notice: raise both by hand.
  On Windows `torch` comes from the `cu128` index. `OMNI_KIT_ACCEPT_EULA=YES` is set by the person running
  Isaac, never by this repo, and the Isaac tests skip without it (the Newton tests skip without an NVIDIA GPU).
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
  `qpos`, float64), which `eval_policy.py --save-dataset` sets; older datasets cannot be played back, and a
  different `nq` is rejected. Both are single-instance only.

## F. `research/` is a folder you fill with a plugin repo; a paper ships as its own repo

- **Not a submodule.** `research/` is git-ignored, like `assets/` and `checkpoints/`. Whatever repository
  you clone into it is the plugin: `physai-study-NN` for a published study, your own lab's repo, or the
  private `PhysAI-Robot/physai-research` that holds every unpublished study. The folder name stays
  `research/` because the plugins import themselves as `research...` (`plugins: [research.plugin]` in a
  manifest), so the run commands do not change from one repo to the next. A submodule pointer to a private
  repository would show every public user a URL they cannot clone and fail `--recurse-submodules` for them.
- **The core works without it.** Research loads as a plugin named in a manifest
  ([G](#g-plugins-load-by-name-from-the-manifest-core-ships-a-demo-policy)); the research-only scripts and
  tests live in the plugin repo. This repository's CI has no checkout of it and holds no credential for it:
  the plugin's own tests run in its repository against a tagged or `dev` core.
- **One public repo per paper.** When a paper is out, publish just that study's runnable code (topic modules,
  manifests, tests; no findings or plans) as `PhysAI-Robot/physai-study-NN`, with checkpoints and data on
  Hugging Face. It is tagged like the core (`study-NN-v1` on both), and its README says to clone the core
  at that tag and the study into `research/`. `physai-research` itself never goes public.
- **Pinning.** A plugin version is a tag in the plugin repo; the matching core version is the same tag
  here. There is no pointer in this repository to update.


## G. Plugins load by name from the manifest; core ships a demo policy

- **Physai is the framework; research is a plugin.** Anyone researches in their own folder with
  their own scenes, tasks and policies. `physai-research` is one such folder, not a special
  one: core has no code path that names it.
- **The manifest is the loader.** An optional `plugins: [module, ...]` list is imported by
  `physai.plugins.load_plugins` before any name is validated, so one manifest is self-contained
  and a run's plugins are recorded with it. There is no `--plugin` flag and no environment
  variable. `--policy` and the other content flags only override what the manifest says, and
  `--policy` is checked against the registries after the manifest has loaded its plugins.
- **A scene variant is data.** A different layout, range or cube is `scene.name` plus
  `scene.overrides` in the researcher's manifest. A genuinely new scene registers itself with
  `register_scene`. Scenes are `ManipulationSceneConfig` today; anything else needs a seam
  extension in core first.
- **Core still moves a robot.** `so101`'s `demo` policy (a planner plus `PlanRunner`, true
  object poses, which looks each cube up again before it descends) is the minimal proof that a
  policy drives the arm and the default of `run_sim.py` and `eval_policy.py`. It plans the
  single cube with `ScriptedPlanner` and the `stack_cubes` task with `StackPlanner`, the one
  core planner that decides something (order, level heights, carry height, and, when the
  arm's top-down grasp is too low for the tower, the tilt and roll of each waypoint). It is a demo, not a
  baseline to beat; a researcher's expert, visual servo or ACT checkpoint comes from their
  plugin. Language grounding stays out of core: `StackPlanner` takes an order, not a sentence.
- **A policy declares its own traits.** `register_policy` and `register_robot_policy` take
  `PolicyTraits` (`needs_images`, `simulators`); manifest validation rejects a policy on an
  engine it does not list, and `eval_policy.py` renders cameras from `needs_images`. A policy
  may also override `episode_metrics()` (extra columns in an evaluation row) and set
  `training_seeds` (evaluation warns when seeds overlap). No core script branches on a
  policy's name, so a plugin's policy is treated like a built-in one.
  `run_sim.py --list-policies` and `eval_policy.py --list-policies` print what is available
  for the manifest's robot once its `plugins:` have loaded.
- **Generic tools live in core, study-specific ones with the study.** Evaluation, recording
  (`--save-dataset`), difficulty sweeps and sharded evaluation (`eval_policy_sharded.py`) work on any
  registered policy and are core scripts. A script that depends on one study's cameras, probes or
  plots stays in that study's repo and reuses core's `scripts/` helpers.

## H. Core does not ship weights; `--checkpoint` takes a path or a Hugging Face reference

- **No weights in the repository.** Model snapshots stay out of git (`AGENTS.md`); a paper's checkpoints
  live on Hugging Face (see [F](#f-research-is-a-folder-you-fill-with-a-plugin-repo-a-paper-ships-as-its-own-repo)),
  one repository with one subfolder per checkpoint.
- **One flag, two forms.** `--checkpoint` is a folder, as before, or `hf:<owner>/<repo>/<subfolder>[@<revision>]`.
  `physai.policy.checkpoints.resolve_checkpoint` downloads only that subfolder into `checkpoints/<owner>/<repo>/<subfolder>`
  at the repository root (git-ignored, like `assets/`, so a checkout shows where its weights are and an
  offline machine can be prepared with `scripts/fetch_checkpoints.py`) and hands the policy the local folder,
  so a policy still receives a path and no plugin changes. It lives in core because `--checkpoint` does, and
  any plugin's policy gets it. Unchanged files are not downloaded again; changing the revision of a folder
  replaces its files in place.
- **Optional dependency, lazy import.** `huggingface-hub` is imported only when a reference is used
  (`uv sync` installs it through lerobot); without it the error says what to install.
- **A revision pins a result.** Without one, `main` is used, which can change; a published number should
  name a commit. The evaluation JSON records the reference as given.

## I. Three engines, and the renderer is chosen apart from the physics

Goal: run one scene on MuJoCo, Isaac Sim or Newton, and draw it with any of the three, so a
policy can be measured against a change of physics, of renderer, or both (`--sim` and `--render`).

- **The MuJoCo model and data are the shared state.** Every SO-101 and ViperX 300 S env keeps its
  scene in a MuJoCo `model`/`data`: the simulation itself on MuJoCo and Newton, the display
  mirror (`_sync_mirror`) on Isaac. A renderer is any object with `render(camera) -> HxWx3 uint8`
  and `close()` that draws from that pair (`MuJoCoRenderer`, `NewtonRenderer`, `IsaacRenderer`);
  `robots/rendering.py:attach_renderer` swaps it onto `env.renderer`, and the factories take
  `simulator` and `renderer` independently (default: the simulator's own). Any of the nine
  pairings works for both robots.
- **Newton physics is MuJoCo's solver on the GPU.** `SO101NewtonEnv` and `ViperX300SNewtonEnv`
  are the MuJoCo envs with `step_simulation` replaced by Newton's `SolverMuJoCo` (`NewtonPhysics`):
  the scene is exported to an MJCF beside the robot's meshes, one control period runs as a CUDA
  graph, and the result is copied back into `data` so sites, kinematics, contacts and cameras keep
  reading MuJoCo. It reproduces MuJoCo to within about 1e-4 rad away from contact and follows it
  through a pick and place. Limits: no `domain_randomization` (it edits the MjModel after Newton is
  built), a free joint's `qvel` is Newton's world-frame velocity, and contacts differ by a pad or
  two where the gripper rests on the table. A policy that runs on MuJoCo runs on Newton
  (`PolicyTraits.runs_on`).
- **Isaac and Newton as renderers of someone else's state.** `NewtonRenderer` builds its own Newton
  model from the scene MJCF with the studio's tiled floor and takes the state, the target pad and
  the camera pose from `data` each frame. Isaac draws another simulator's state through
  `show_state`, which writes the joint positions, the cube and the target into its stage and steps
  once (a pose written through the tensor API reaches the cameras only after a step). Visual
  randomization (`domain_randomization`, lighting scale, camera jitter) belongs to one engine's
  scene, so a renderer other than the simulator refuses it.
- **`--video` records a diagonal view, not an observation camera.** The ViperX `front` camera
  is a task view with the arm cut off at the top edge, and a policy reads it, so it stays as it is.
  Instead `WorkspaceConfig.video_view` (a `ViewCamera`, MuJoCo's free-camera parameters, sized per
  robot in `scene_defaults`) is a free camera that every renderer draws on request under the name
  `video`. It is not in any model, so it is no observation and no camera-randomization draw (adding
  a model camera would shift every seeded jitter); `VideoObserver` renders it each step, and
  `--camera front|wrist` records what a policy sees. File names end `_o` (success) or `_x` (failure).
- **Same limits as Isaac (D).** Single-cube scenes for an Isaac simulator or renderer; no `world`
  block, `backend: ros2_sim` or `viewer.mode` but `none` with a non-MuJoCo simulator or renderer;
  `--viewer`, `SharedWorld`, the ROS2 bridge and TurtleBot stay MuJoCo-only. With Isaac in the pairing
  `--serve` runs the `Host` on the main thread.
- **One environment.** Newton is a default dependency, pinned to the version Isaac Sim 6.2.0.0 pins
  (`newton==1.6.1`, `mujoco-warp==3.12.0`, `mujoco==3.12.0`), so it and Isaac share one environment and one lock
  with no conflict and nothing overridden for them. Newton 1.7 (mujoco-warp 3.15) was tried and would have split
  them into two environments; following Isaac is what keeps one. Study 01 ran MuJoCo 3.13, so its MuJoCo numbers
  are not reproduced here, while its Newton scores (1.6.1) are. One ordering rule: Kit brings its own warp,
  and an Isaac app started after the pip warp has been imported comes up without `omni.usd`, so the
  factories start Isaac first (`start_isaac_first`), and nothing may import `newton` or `warp`
  at collection or import time (hence `find_spec`, not `importorskip`, in the Newton tests).
- **Contact settings carry over to Newton by hand.** `SolverMuJoCo.register_custom_attributes` is called
  before the MJCF is read, or the pads' `condim`, `solref` and `solimp` are dropped. MuJoCo's writer omits
  trailing default numbers (`solref="0.004"`) and Newton reads them as zero, so `scene_mjcf` writes them
  out. mujoco-warp has no no-slip solver (`noslip_iterations > 0` raises), which is what holds the SO-101's
  cube in MuJoCo, so Newton runs `cone="elliptic"` with `impratio` 20 (the description's `sim_overrides.newton`
  may change it). Without a substitute the SO-101's cube creeps 15 mm in 12 s. The SO-101 sets `impratio` 30
  (the scripted pick starts to fail from about 100) and, on its pads, `condim` 6 with torsional and rolling
  friction (they stop the cube turning in the jaws) and a contact damping ratio of 2; the cube then moves
  at most 0.7 mm in 60 s. The hold is measured after a 10 s settle, because the cube turns and seats itself
  between the jaws as they take its weight (up to about 2.5 mm, in the first seconds) and then stops;
  `tests/core/acceptance/test_held_cube.py` checks every robot and engine at under 2 mm over the next 60 s
  (SO-101 MuJoCo 0.01 mm, SO-101 Newton 0.7, ViperX MuJoCo 1.3, ViperX Newton 0.3, both on Isaac Sim 0.00).
- **The ViperX on MuJoCo integrates implicit-fast, with stiffer friction and a damped contact.** Its arm
  actuators are soft and damped, and with the default Euler integrator the held arm trembles and shakes a held
  cube loose (1.5-3 mm in 60 s); `implicitfast` stops that, as in Newton's solver. What is left is MuJoCo's soft
  friction, which the stock gripper (about 1 N per side) does not hold against: `impratio` 30 and a contact
  damping ratio of 3 on the pads and the cube bring the drift to at most 1.4 mm over ten seeds (60 s after a
  10 s settle; 1.3 mm in the test's four). Damping the pads alone is worse (2.9 mm), because MuJoCo averages the two geoms' `solref`.
  No-slip is not the fix here: it drops the ViperX's cube.
