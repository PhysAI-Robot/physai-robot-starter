# 9. Record and replay web viewer sessions through the existing data path

## Status

Accepted.

## Context

The web viewer (`--serve`) supports live jogging and monitoring, but a
session cannot become research data: nothing in `physai.web` records, and
recorded episodes can only be inspected offline. Demonstrations are
currently written only by `scripts/collect_demos.py` through
`physai.data.EpisodeRecorder`, whose `.npz` layout uses LeRobot-shaped
feature keys (`observation.state`, `action`, `observation.images.<cam>`,
...) plus a `meta.json` with a per-episode `success` flag.

Recording from the browser and stepping back through a recorded episode
both need new operations on `Host`, whose method surface is frozen (see
[Frozen vs. free to change](../ARCHITECTURE.md#frozen-vs-free-to-change)).
Frame-accurate playback also needs more state than an episode stores today:
`observation.state` holds only the robot's joint positions, so object poses
(e.g. the cube) cannot be restored from it.

## Decision

Extend `Host` additively; rename and remove nothing:

- A `record_dir` option on `Host.for_robot()`, plus `start_recording()`,
  `stop_recording(success)` and `recording_status()`. `success` is `True`,
  `False`, or `None` (discard). Recording writes through
  `EpisodeRecorder`; the web layer never defines its own episode format.
  The recorded action is the resolved joint-target action the robot actually
  stepped with, not the raw browser twist. Steps are skipped until every
  camera has produced a frame, and a world reset or a policy ending its
  episode discards the take in progress, so a take never mixes worlds or
  holds steps with missing images.
- `list_episodes()`, `load_episode(file)`, `seek(frame, relative=False)`,
  `set_playback(playing, speed)`, `exit_playback()`. Playback exists only
  while the world is paused: it restores recorded simulator state and calls
  `mj_forward`, and it is advanced from the paused branch of the existing
  physics loop by measured wall time, so 1x is real time even if the loop
  runs slower than `control_hz`. There is no second playback engine and no
  re-simulation. Only episodes the dataset lists can be loaded. Entering
  playback snapshots the live world; exiting restores it and stays paused.
  Commands, resets, resuming and recording are rejected while playback is
  active.
- Matching WebSocket messages (`record_start`, `record_stop`,
  `playback_load`, `playback_seek`, `playback_step`, `playback_play`,
  `playback_exit`) and one read-only route, `GET /api/episodes`, which lists
  the record directory's `meta.json` episodes.

Add one optional per-step key to the episode `.npz`:
`observation.environment_state`, the full MuJoCo `qpos` (float64), named after
LeRobot's feature-key convention. `EpisodeRecorder` gains an optional
`environment_state_dim` constructor argument and an `environment_state`
argument on `record()`; when the dimension is set, every step must supply the
state and `meta.json`'s `features` declares the key. Without it the recorder
behaves exactly as before, so datasets recorded before this change keep
their key set and cannot be played back in the viewer. `collect_demos.py`
sets the dimension and records the state, so new scripted datasets are
playable. Continuing an
existing dataset requires the same robot and the same `environment_state`
layout.

The state snapshot gains `paused`, `recording`, and `playback` fields, merged
in by `Host.latest_state()` at read time so they stay current while the world
is paused. The snapshot keeps `"version": 1`, because existing fields are
neither renamed nor removed.

Recording and playback are limited to a single-instance `Host`
(`Host.for_robot`). A shared-world `Host` rejects them with a clear error
until there is a concrete need for per-instance datasets.

## Consequences

A browser session produces the same dataset shape that `collect_demos.py`,
`eval_policy.py`, and the research training code already consume, with a
per-episode success tag. No parallel action contract appears, so ARCHITECTURE's
data-format rule still holds. Because every change adds something new and
existing clients keep working, no deprecation window is needed.
Playback is tied to the model that recorded the episode: an episode whose
`environment_state` width differs from the current `model.nq` is rejected.
Playback shows the recorded geometry and recomputes kinematics (the tip
pose) from it, but contact force is reported as unavailable: restoring `qpos`
cannot reproduce the actuator state behind the original squeeze force.
Scripted demonstrations are replayable in the viewer because
`collect_demos.py` passes `environment_state`; datasets collected before that
change are not.
