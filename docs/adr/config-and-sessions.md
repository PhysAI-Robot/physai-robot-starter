# Configuration and session decisions

ADRs 2 and 10. The manifest's field list and validation rules are in
`physai/config/manifest.py`'s docstring; the summary is in
[ARCHITECTURE.md](../ARCHITECTURE.md#session-manifest).

## ADR 2: One session manifest schema

Status: accepted. Extended by ADR 10.

**Context.** Configuration was split across three schemas that could not
describe "N robots, a task, a policy, a backend and viewer options" together:
`configs/sim_config.yaml` (seed and randomization), `configs/tasks/<robot>/*.yaml`
(one robot, task, scene, env) and `configs/worlds/*.yaml` (placement only).

**Decision.** Add `SessionManifest` (`physai/config/manifest.py`, files under
`configs/manifests/`): `robots` (`{id, robot, pose, task?, policy?}`), a scene, a
`backend` (`direct | ros2_sim | ros2_real`) and `viewer` options. A single-robot
run is a manifest with one entry. The old loaders stayed at first.

**Consequences.** Multi-robot, multi-backend sessions fit in one file.
`ros2_real` is accepted for forward compatibility and raises "not yet
implemented" (simulation only for now, see [ROADMAP.md](../../ROADMAP.md)).

## ADR 10: The manifest becomes the run description

Status: accepted. Extends ADR 2.

**Context.** Nothing consumed the manifest. `scripts/run_sim.py` still had three
input paths (`--config`, `--world`, bare `--robot`), each assembling objects by
hand and branching on robot names, and the same defaults were typed in several
places (the seed had a three-level fallback; the success tolerance and hold
length lived in four files).

**Decision.**
- Extend the schema additively (`schema_version` stays 1): a `world` block,
  per-robot `model` and `config`, `success_hold_steps`, and normalisation of
  overrides and paths. Several robots imply a shared world.
- Add `physai.runtime.create_session()`: one robot becomes a `create_runtime()`
  composition, a `world` becomes a `SharedWorld` (no task or policy yet).
- `physai.config.compat` converts the old inputs, and every `run_sim.py` run is
  built from the resulting manifest. `--config` and `--world` print a
  deprecation notice and are removed after a deprecation window.
- One source per default: the `simulation` block owns seed and randomization (a
  robot `config` repeating them is rejected), the task and `TaskRuntime` own
  success defaults, the robot registry owns a robot's default task, and a robot's
  capabilities decide the shape of the `constant` policy.
- A host-driven session steps the bare robot and composes no task (a task would
  end episodes on success; it only picks the scene). The host's camera thread
  renders only for robots that accept `camera_stride=0`; TurtleBot4 renders
  inline because a second GL context on another thread fails on Windows.

**Consequences.** One representation describes every run, and a new robot needs
no branch in `run_sim.py`. Other scripts still assemble robots by hand and
shared-world sessions still run no task or policy; both are listed under
[Known remaining gaps](../ARCHITECTURE.md#known-remaining-gaps).

## ADR 20: `run_sim` and `eval_policy` share one session and one rollout

Status: accepted. Narrows ADR 10's "other scripts still assemble robots by hand" for
`eval_policy.py`.

**Context.** Both scripts run episodes, so they had drifted: `eval_policy.py` built its
env and task from flags and its own defaults (600 steps against the manifest's 400 at the time)
while `run_sim.py` read a manifest, and the episode loop was written twice. The copies
disagreed on `policy.done` (ignored by evaluation), on a refused action (a crash in
`run_sim`, an `unsafe_action` result in evaluation), on the Isaac render check and on the
default camera and video folder.

**Decision.** The two scripts keep different purposes (`run_sim.py` to run and look,
`eval_policy.py` to measure; see
[Script roles](../ARCHITECTURE.md#script-roles)) and share the rest.
1. The session comes from a manifest. `eval_policy.py` loads one
   (`so101_single_cube_fixed_place.yaml`, or the new `so101_sorting.yaml` for
   `--sorting`) and turns its difficulty flags into overrides through
   `config.compat.with_overrides` (`domain_randomization`, `scene_overrides`,
   `robot_config`, which carries Isaac's lighting and camera-jitter knobs).
2. `physai.runtime.run_episode` owns the loop: it ends an episode on termination,
   truncation, `policy.done` or a `SafetyViolation` (recorded in the result), and raises
   `RenderGlitch` when Isaac stops drawing the robot. Video and recording are
   `EpisodeObserver`s.
3. `max_steps` has no script default: it is the manifest's, and `--max-steps`
   overrides it. The output flags are defined once (`--video`, `--camera`, `--record`,
   `--out`, `--name`) and `scripts/_outputs.py` writes them.
4. `collect_demos.py` builds its session from the same manifests and runs episodes
   through `run_episode`, so it no longer assembles an environment by hand
   (`eval_randomization.py` was dropped: two `eval_policy.py` runs, with and without the
   difficulty flags, compare the same thing). `run_sim.py` drops its deprecated `--config` and `--world`
   inputs (and `compat.manifest_from_task_file`/`manifest_from_world_file`, with
   `configs/worlds/`); `--policy-arg` and `--checkpoint` are shared.

**Consequences.** The manifest's `max_steps` is now 600, so evaluation keeps its old length
and `run_sim.py` runs 600 steps too (it was 400). Every successful episode in the
existing ACT, visual servo and scripted results ended within 353 steps, so 600 is margin;
failures run the full length. The research READMEs and `sweep_difficulty.py` still pass
`--max-steps 600` explicitly.
`eval_policy.py` lost `--robot` and `--render` (the manifest names the robot; cameras
render whenever a video, recording or image policy needs them), and `--video-dir` and
`--video-name` became `--out` (videos under `videos/`, recordings under
`recordings/`) and `--name`; later flag renames are in ADR 21. `collect_demos.py` also loses `--robot`, and its dataset now
carries the `extras.*` arrays; its states, actions and rewards are unchanged.

## ADR 21: One flag name per concept across the scripts

Status: accepted. Supersedes the flag names in ADR 20's consequences.

**Context.** The scripts had grown their own names for the same things: the dataset
folder was `--out` (`collect_demos`), `--dataset-dir` and `--dataset`; the output
location was `--out` (a folder in one script, a file in two), `--out-dir` (a folder of
videos in `run_sim`/`eval_policy`, a folder of results in `run_sharded_eval`/
`sweep_difficulty`, so the two collided when one passed arguments to the other),
`--dest`, `--save-plan` and `--save-frames`; results went to `--json-out` or
`--merged-out`; the episode limit was `--max-steps` or `--max-ticks`. 45 flags had no help.

**Decision.** One short name per concept (the table in
[CLI conventions](../ARCHITECTURE.md#cli-conventions)): `--out DIR` for the folder a
script writes into (`--out FILE` for the two single-file tools), `--json FILE` for a
result, `--dataset DIR` for a dataset folder, `--max-steps` everywhere. Every script builds
its parser with `scripts/_cli.new_parser`, so `--help` has one layout and shows defaults,
and `tests/core/unit/test_cli_conventions.py` rejects the old names and flags without help.
The old names are removed, not aliased.

**Consequences.** Headless `run_sim.py` and `eval_policy.py` no longer write the dataset
layout (`--dataset-dir` is gone); `--record` writes one file per episode, and a dataset
comes from `collect_demos.py` or from recording in the browser (`run_sim.py --serve
--dataset`). `run_sharded_eval.py` passes its `--out` to every shard, so their videos
and recordings land in the result folder. `sweep_difficulty.py` takes `--sim`; on Isaac it
runs only the cells `eval_policy.py` supports there.

## ADR 22: The manifest is the only run description

Status: accepted. Finishes the migration ADR 10 started; supersedes its note that a bare
`--robot` and the task files stay as conversions.

**Context.** After ADR 20 every run script read a manifest except two paths kept for
back-compat: `run_sim.py --robot/--sim-config` (a manifest built from a robot name and
`configs/sim_config.yaml`) and `run_ros2_sim.py --config` (a task YAML read by
`load_task_config`). They needed a second loader module (`config/legacy.py`), the
`configs/tasks/` and `configs/sim_config.yaml` files, `compat.manifest_for_robot` and their
own tests, to describe what a manifest already says.

**Decision.** Remove them. `run_sim.py` takes `--manifest` (default: the SO-101 single-cube
session) and TurtleBot4 gets `configs/manifests/turtlebot4.yaml`. `run_ros2_sim.py
--manifest` builds the node's environment with `physai.runtime.robot_env_config`, which
reuses the manifest's robot fields and scene exactly as `create_session` does (a test
compares the two). `SimulationConfig` and its parser now live in `config/manifest.py`;
`config/legacy.py` is gone. The same pass dropped `scripts/eval_randomization.py` (two
`eval_policy.py` runs cover it), the empty `TaskBackend` protocol and the unused `lark`
dependency.

**Consequences.** `run_sim.py --robot X` and `run_ros2_sim.py --config FILE` are gone; the
runbooks use manifests. The ROS2 node itself is only exercised where ROS2 is installed; the
config it receives is checked against `create_session` on MuJoCo.
