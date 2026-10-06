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
env and task from flags and its own defaults (600 steps against the manifest's 400)
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
   overrides it. The video flags are defined once (`--video`, `--video-dir` with `--out`
   as an alias, `--camera`).

**Consequences.** The evaluation default is 400 steps, not 600; runs that must keep 600
pass `--max-steps 600`, as the research READMEs and `sweep_difficulty.py` already do.
`eval_policy.py` lost `--robot` (the manifest names the robot). `collect_demos.py` and
`eval_randomization.py` still assemble their own environments.
