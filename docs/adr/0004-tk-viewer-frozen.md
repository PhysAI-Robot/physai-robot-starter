# 4. Native MuJoCo viewer, not a maintained Tk client

## Status

Accepted.

## Context

There are two client UIs: a desktop viewer (`--viewer`) and a FastAPI +
Three.js web viewer (`--serve`). If both grow features, every new capability
has to be built and maintained twice. The desktop side was originally a
hand-rolled Tk client (`SingleWindowViewer`): MuJoCo frames blitted into Tk
`PhotoImage` widgets as raw PPM bytes every tick, with custom mouse-drag
orbit/zoom and a hand-styled toolbar reimplementing what a real viewer
already provides — about 240 lines with no functional payoff over what
MuJoCo ships for free. Its one feature the web viewer didn't already have
was a live multi-camera sidebar; the web viewer has since grown its own,
better one (`src/physai/web/static/js/cameras.js`: add/remove slots,
persisted layout).

## Decision

`--viewer` opens MuJoCo's own `mujoco.viewer.launch_passive`, not a
maintained Tk client. Its scope stays frozen at simulation render and basic
status; it never grows features (multi-camera panels, jog controls, instance
selection, telemetry, overlays) — those go into the web viewer only, per
`scripts/teleop_keyboard.py`'s existing `launch_passive` pattern.
`Host.start()` already steps physics on its own thread; the viewer renders a
private `MjData` copy refreshed each tick under `host.physics_lock`,
mirroring `Host._camera_loop`'s pattern — the native viewer's own render
thread touches that copy continuously, not only at `sync()`, so sharing
`host.data` directly races Host's physics/camera threads against it.

## Consequences

There is no Tk-specific render loop, PPM-blit code, or `tkinter` runtime
dependency to maintain. `--viewer`'s scene navigation and rendering quality
come from MuJoCo's own viewer instead of being hand-maintained. Reviewers can
still reject any PR that tries to add UI features to `--viewer` on scope
grounds alone — the web viewer remains the one place sophistication
accumulates.
