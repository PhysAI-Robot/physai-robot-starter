# Web host and viewer decisions

ADRs 1, 4 and 9. The host's API surface is in
[ARCHITECTURE.md](../ARCHITECTURE.md#host--client-api); the operating guide is
[WEB_VIEWER_RUNBOOK.md](../WEB_VIEWER_RUNBOOK.md).

## ADR 1: One `Host` for single-robot and shared-world sessions

Status: accepted.

**Context.** `SimulationHost` (one robot) and `SharedWorldHost` (many) each
implemented their own physics thread, camera worker, control lease and publish
loop, and `web/app.py` branched on `isinstance`. A host feature had to be
written twice.

**Decision.** One `Host` class (`src/physai/web/host.py`); a single robot is a
world with one instance. Every method (`scene`, `latest_state`, `list_robots`,
`camera_jpeg`, `submit`, `reset`, `set_paused`, `release_control`) is
instance-keyed, and the id may be omitted when there is one robot. Reset and
pause stay world-atomic.

**Consequences.** `web/app.py` never branches on host type. The old
`web/runtime.py` and `web/world_runtime.py` were removed after tests covered
both the one-instance and many-instance cases on the same class.

## ADR 4: `--viewer` is MuJoCo's own viewer, frozen in scope

Status: accepted.

**Context.** A hand-rolled Tk client (about 240 lines of PPM blitting, custom
orbit and zoom) duplicated what MuJoCo's viewer ships, and every UI feature
would have had to be built twice (desktop and web).

**Decision.** `--viewer` opens `mujoco.viewer.launch_passive` and stays limited
to rendering and basic status. Camera panels, jog controls, instance selection,
telemetry and overlays go into the web viewer only. The viewer renders a
private `MjData` copy refreshed each tick under `Host.physics_lock`, because
its render thread touches the copy continuously and sharing `host.data` would
race the physics and camera threads.

**Consequences.** No Tk code or dependency remains. A pull request that adds UI
features to `--viewer` can be rejected on scope alone.

## ADR 9: Record and replay web sessions through the existing data path

Status: accepted.

**Context.** Browser sessions could not become research data, and recorded
episodes could only be inspected offline. Frame-accurate playback also needs
object poses, which `observation.state` (joint positions only) does not hold.

**Decision.** Extend `Host` additively, removing nothing:
- `record_dir` on `Host.for_robot()`, with `start_recording()`,
  `stop_recording(success)` (`True`, `False`, or `None` to discard) and
  `recording_status()`. Recording goes through `EpisodeRecorder`; the web layer
  has no episode format of its own. The recorded action is the resolved joint
  target the robot stepped with. Steps wait until every camera has produced a
  frame, and a world reset or a policy ending its episode discards the take.
- `list_episodes()`, `load_episode(file)`, `seek()`, `set_playback()`,
  `exit_playback()`. Playback exists only while paused: it restores recorded
  state and calls `mj_forward`, advanced by wall time from the existing physics
  loop, with no re-simulation. Entering snapshots the live world and exiting
  restores it, still paused; commands, resets, resume and recording are refused
  meanwhile.
- WebSocket messages (`record_start`, `record_stop`, `playback_load`,
  `playback_seek`, `playback_step`, `playback_play`, `playback_exit`) and one
  read-only route, `GET /api/episodes`.
- One optional per-step key in the episode `.npz`,
  `observation.environment_state` (full `qpos`, float64), enabled by the
  recorder's `environment_state_dim`. `collect_demos.py` sets it. The state
  snapshot gains `paused`, `recording` and `playback`, merged at read time, and
  keeps `"version": 1`.
- Recording and playback are single-instance only; a shared-world host rejects
  them until per-instance datasets are needed.

**Consequences.** A browser session yields the same dataset shape as
`collect_demos.py` and `eval_policy.py`, with a per-episode success tag. Nothing
was renamed, so no deprecation window was needed. Playback is tied to the model
that recorded the episode (a different `nq` is rejected), recomputes the tip
pose, and reports contact force as unavailable because `qpos` cannot reproduce
the original squeeze. Datasets recorded before `environment_state` existed
cannot be played back.
