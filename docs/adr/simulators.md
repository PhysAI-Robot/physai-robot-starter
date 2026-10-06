# Simulator engine decisions

ADRs 13 to 16 and 18: how Isaac Sim joined MuJoCo as a second engine. Read in order;
each later ADR supersedes part of the earlier one.

## ADR 13: Isaac Sim as an optional backend

Status: accepted. The package path, the adapter name, the install recipe and the
manifest wiring were superseded by ADRs 14, 15 and 16; the robot-only scope of
`SO101IsaacEnv` was relaxed by ADR 15's same-scene work.

**Context.** The goal includes cross-simulator portability: evaluating a policy
tuned in MuJoCo against a second engine to measure the sim-to-sim gap. A spike
on Isaac Sim 6.1.0.0 (Python 3.12, RTX 3060) verified URDF import, physics
variant selection, actuator gain mapping, closed-loop joint control and camera
rendering against the real engine.

**Decision.** An Isaac package (SimulationApp lifecycle, `RobotDescription` to
USD, world extras) and `robots.so101.isaac_env.SO101IsaacEnv`, mirroring the
MuJoCo side. Only those two locations may import `isaacsim`, `omni` or `pxr`
(import-linter), and every such import is function-local because Isaac's modules
are not import-safe before a `SimulationApp` exists, so importing the packages
never needs Isaac installed. No new registry seam: the direct adapter wraps any
`RobotPort`, so it wraps an Isaac-backed port unchanged. `make_so101()` gained a
`simulator` parameter that picks which port it builds. `ArmKinematics` stays a
MuJoCo-specific model, reused by Isaac as a kinematics oracle only, because
changing its API touches 25+ call sites with no second consumer to validate it.

**Consequences.** Isaac's USD variant mechanism (`mujoco`, `physx`, `physics`,
`none`) makes a more direct sim-to-sim bridge than assumed; `physx` is the
default and `mujoco` stays available for comparing solvers.

## ADR 14: Simulator package layout and adapter names

Status: accepted.

**Context.** `physai.sim` held MuJoCo code under a name that did not say so, with
Isaac as a flat sibling. The transport adapters `DirectMuJoCoAdapter` and
`ROS2MuJoCoAdapter` never contained MuJoCo code, and their registered names did
not match the manifest's `backend` values.

**Decision.** Both engines are peer subpackages of `physai/sim/`:
`mujoco/` (core, shared world, domain randomization, scenes) and `isaac/`, with
`sim/__init__.py` holding only a docstring so importing `physai.sim` never needs
an engine SDK. `robots/so101/env.py` became `mujoco_env.py` beside
`isaac_env.py`. Adapters are renamed to match `backend` one to one:
`DirectAdapter` (`direct`), `ROS2SimAdapter` (`ros2_sim`), `ROS2HardwareAdapter`
(`ros2_real`). No compatibility aliases were kept; every import site was updated
in the same change. Two import-linter rules were added: the two engine packages
must not import each other, and `physai.sim.isaac` must not import `mujoco`.

**Consequences.** This touches a frozen boundary (see ARCHITECTURE's "Frozen vs.
free to change"). One subpackage per engine is the pattern a third engine
follows. An old adapter name now fails with the usual "unknown adapter" error
listing the current names.

## ADR 15: Simulator engine selection

Status: accepted.

**Context.** The engine was reachable only through `make_so101`'s `simulator`
argument, in turn only through a manifest robot's free-form `config:`. Nothing
checked that a robot supported the engine, `_robot_fields` filtered config keys
against MuJoCo's `EnvConfig` even for an Isaac build, and there was no command
line switch.

**Decision.**
- `RobotDescriptor.simulators` (default `("mujoco",)`; so101 declares
  `("mujoco", "isaac")`). `create_robot` validates `simulator` against it, and
  `available_simulators(name)` exposes the check. `create_env_config` forwards
  `simulator` only for robots that declare more than one, via `so101_env_config()`
  (`EnvConfig` or `IsaacEnvConfig`). `create_runtime` takes `simulator`.
- The manifest has an optional top-level `simulator` (default `mujoco`),
  independent of `backend`. Validated at load: every robot must support it, a
  robot's own `config` must not repeat it, and a non-MuJoCo engine rejects a
  `world` block (or several robots), `backend: ros2_sim` and any `viewer.mode`
  other than `none`.
- `run_sim.py --sim {mujoco,isaac}` overrides it through
  `config.compat.with_overrides`, which re-runs the registry and cross-field
  checks via `validate_manifest()` (a bare `replace` would skip them). `--sim
  isaac` with `--viewer`, or `--serve` with `--record-dir`, is refused up front.
- **Isaac `--serve`.** The web `Host` runs on the main thread (`Host.run()`,
  because Isaac must stay on the thread that created `SimulationApp`) and uvicorn
  on a worker thread with its own event loop (Kit replaces `asyncio.run`).
  `SO101IsaacEnv` exposes a display mirror (the kinematics-oracle model plus an
  `MjData` refreshed from Isaac's joint state and cube pose, never simulated); with
  a scene the oracle is the scene's MuJoCo model, so the view shows table, target
  and cube. No Cartesian jog, no recording or playback.
- **Same scene on both engines.** `IsaacEnvConfig.scene` takes the scene config
  MuJoCo builds from, so table, cube, target, front camera and the per-seed cube
  position (drawn through `layout.draw_cube_xy`, pinned by `golden_layouts.json`)
  have one source. `SO101IsaacEnv` exposes `table_top` and `cube_half` so
  `TaskRuntime` wraps it, and `eval_policy.py --sim isaac` runs the same scene,
  seeds and report. `collect_demos.py` does not, because the scripted expert reads
  MuJoCo state directly. Limits: single-cube scenes, fixed target, no domain
  randomization beyond a lighting scale and camera jitter, observation-only
  policies (`visual_servo`, `constant`).
- `SO101Env.robot_spec.metadata` gains `"simulator": "mujoco"`, so a recorded
  dataset names its engine.

**Consequences.** A bad robot, engine or combination fails at load with a specific
message. `--viewer`, web recording and playback, `SharedWorld`, the ROS2 bridge and
`ArmKinematics` remain MuJoCo-only; this ADR makes that explicit rather than
removing it (see [Known remaining gaps](../ARCHITECTURE.md#known-remaining-gaps)).
`configs/manifests/so101_isaac.yaml` is the robot-only smoke manifest (`policy:
constant`).

## ADR 16: `isaacsim` as a project extra, not a separate venv

Status: accepted. Replaces ADR 13's separate-venv recipe.

**Context.** ADR 13 ruled out an extra because `isaacsim==6.1.0.0` pins
`numpy==2.3.1` against this project's `numpy>=2.0,<2.3`, and because it needs an
NVIDIA index and pre-releases. The numpy ceiling actually came from
`lerobot==0.6.1` (`numpy<2.3.0`), not MuJoCo.

**Decision.** `isaac = ["isaacsim[all,extscache]==6.1.0.0"]` is a normal extra:
`uv sync --extra isaac`. Found by iterating on real `uv lock` errors:
1. An `nvidia` index (`https://pypi.nvidia.com`), because the ecosystem is not on
   PyPI.
2. `index-strategy = "unsafe-best-match"` (project-wide, not Isaac-scoped; low risk
   given the NVIDIA-specific names).
3. `fastapi` and `uvicorn[standard]` moved into base dependencies with a loose
   floor (`uvicorn>=0.29`), since `isaacsim-kernel` pins `uvicorn==0.29.0` and uv
   resolves each fork separately.
4. A `webtest = ["httpx2"]` extra marked in conflict with `isaac` (`httpx2` needs
   `idna>=3.18`, isaacsim pins `idna==3.10`); it is used by one test already
   guarded by `importorskip`.
5. The old `dev` extra (`pytest`, `ruff`, `lark`, `import-linter`) moved into base:
   this is a repository worked in directly, not a library.
6. `vla` (`lerobot`, `numpy<2.3`) is marked in conflict with `isaac` instead of
   dropping `lerobot` or installing it by hand.

`training` and `isaac` combine freely; `webtest` and `vla` are narrow and
genuinely incompatible. `mujoco` and `numpy` resolve twice in the lockfile (the
Isaac fork gets `mujoco==3.11.0`, `numpy==2.3.1`); `physai.sim.mujoco` only assumes
`mujoco>=3.2`.

**Consequences.** Verified end to end: `uv sync --extra isaac --extra training`
installs cleanly, the suite passes under it, and a headless episode ran with
`OMNI_KIT_ACCEPT_EULA=YES` set by the person running it (nothing sets it for
them). Bugs found along the way: the headless default policy (`scripted`) needs
MuJoCo-only `ArmKinematics`, so `run_episodes()` now raises a clear error and
`so101_isaac.yaml` sets `policy: constant`; `uv sync` leaves a stale empty
`isaacsim/` namespace package when the extra is dropped, so the Isaac tests check
for `SimulationApp`; and `lerobot`'s `draccus` installs a top-level `tests`
package that shadowed this suite's absolute imports, fixed by adding
`tests/__init__.py` and writing `from tests.conftest import ...`.

## ADR 18: One environment for Isaac Sim and LeRobot

Status: accepted. Supersedes item 6 of ADR 16 (the `vla` extra marked in conflict
with `isaac`).

**Context.** `lerobot==0.6.1` requires `numpy<2.3.0` and `packaging<26.0`, while
`isaacsim-kernel` pins `numpy==2.3.1` and `isaacsim-core` pins `packaging==26.0`,
so ADR 16 kept the two in separate environments and an ACT checkpoint could not
run inside the Isaac process. That blocked the sim-to-sim comparison of a learned
policy. The Isaac environment in use had already drifted to `numpy` 2.2.6 and
`packaging` 25.0, and its Isaac tests still passed.

**Decision.**
1. `vla` is folded into `training` (`gymnasium`, `torch`, `torchvision`,
   `lerobot`), so LeRobot is a training dependency and the extra is no longer
   named after a model family ACT does not belong to.
2. `[tool.uv] override-dependencies` forces `numpy>=2.0,<2.3` and
   `packaging>=24.2,<26.0`, and the `isaac`/`vla` conflict is removed
   (`isaac`/`webtest` stays).
3. On Windows, `torch` and `torchvision` come from the explicit PyTorch `cu128`
   index, because PyPI's Windows wheels are CPU-only.
4. `eval_policy.py --sim isaac` accepts `--policy lerobot`.

**Consequences.** `uv sync --extra isaac --extra training` gives one environment, and
the same checkpoint runs on both engines. Verified in it: the non-Isaac suite
(245 passed, 3 skipped), `lint-imports` (8 contracts kept), each Isaac test file in
its own process (17 tests), and 100 closed-loop ACT episodes on Isaac
([FINDINGS](../../research/imitation_learning/FINDINGS.md)). Not verified: ACT
*training* in this environment, and `uv sync --locked --extra training` on Linux
(CI and Docker, which now also download torch and LeRobot).

The overrides break pins the upstream packages declare, and `uv pip check` lists
them (`numpy`, `packaging`, `setuptools`, and also `pywin32` and `onnxruntime-gpu`,
which no check here exercises). A later `isaacsim` or `lerobot` release may need
`numpy` 2.3, at which point the overrides go and this ADR is superseded. LeRobot's
`opencv-python-headless` and Isaac's OpenCV both write a `cv2` package; the import
works, but only the calls this repository makes were exercised.
