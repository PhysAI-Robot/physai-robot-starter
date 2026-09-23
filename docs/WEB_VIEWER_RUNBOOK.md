# Web Viewer

This is the operational runbook: how to run the viewer, keyboard/control-pad
mapping, appearance, cloud workspace setup, and troubleshooting. The host's
API surface, threading model, and control-lease contract are architecture
reference material and live in
[docs/ARCHITECTURE.md](ARCHITECTURE.md#host--client-api) instead.

The web viewer is a client of an already running simulation host. The host
owns the robot, task configuration, seed, policy, MuJoCo model, physics clock,
and command arbitration. The desktop viewer and browser render the same state
and send control intents to that one host.

Install the web and training dependencies once from the project root:

```bash
uv sync --extra web --extra training
```

After that setup, the commands below use plain `uv run`.

## Start One Authoritative Simulation

Start an idle SO-101 simulation with the browser server. With no explicit
`--policy`, the host holds its current pose and waits for browser jog commands;
it does not start pick-and-place. The desktop viewer is optional; add
`--viewer` when you want both clients attached to the same host:

```bash
MUJOCO_GL=egl uv run python scripts/run_sim.py --robot so101 --serve --seed 0
```

To run the scripted pick-and-place policy instead, make it explicit:

```bash
MUJOCO_GL=egl uv run python scripts/run_sim.py --config configs/tasks/so101/pick_place.yaml --policy scripted --serve --seed 0
```

The desktop GUI and web server now use the same MuJoCo engine. Open
[http://127.0.0.1:8000/](http://127.0.0.1:8000/), or ask the client launcher to
open it:

```bash
uv run python scripts/run_web.py --connect http://127.0.0.1:8000
```

Use another bind address or port on the host when needed:

```bash
MUJOCO_GL=egl uv run python scripts/run_sim.py --config configs/tasks/so101/pick_place.yaml --viewer --serve --host 0.0.0.0 --port 8004
```

The `--robot`, `--config`, `--seed`, and `--policy` options belong to the
simulation host. `run_web.py` does not select a robot or create a simulation.
For interactive viewer sessions, the host reuses the supplied `--seed` on each
automatic reset. A policy is reset only when one was explicitly supplied.

### Shared-world mode

To place multiple heterogeneous robot instances in one MuJoCo scene and clock,
use the world manifest:

```bash
MUJOCO_GL=egl uv run python scripts/run_sim.py --world configs/worlds/heterogeneous.yaml --serve
```

The manifest assigns each instance an ID, robot adapter, model path, and world
transform. It uses a 30 Hz control rate and starts idle. The browser renders
the complete scene. Selecting `arm_1` or `base_1` changes only which instance
receives keyboard commands; it does not start another simulation or remove the
other robot. Reset and pause affect the whole world.

## Run Without A Desktop Window

On a server, in a container, or in any environment with no display, drop
`--viewer` and pass only `--serve`. The web server starts exactly as above,
but no desktop window is opened:

```bash
MUJOCO_GL=egl uv run --extra web python scripts/run_sim.py --config configs/tasks/so101/pick_place.yaml --policy visual_servo --serve --seed 0
```

The host runs until `Ctrl+C` or `SIGTERM`, then stops the web server and the
physics thread in order. `SIGTERM` is handled like `Ctrl+C`, so a container or
process manager stop request takes the same path. Use `MUJOCO_GL=osmesa` when
the machine has no GPU or EGL device.

The host has no authentication. Binding it to a non-loopback address with
`--host 0.0.0.0` lets anything that can reach the port send control commands, so
keep such a port on a private network or behind an authenticated tunnel.

## Cloud Workspace (GitHub Codespaces)

`.devcontainer/devcontainer.json` describes a Python 3.12 container for a
cloud workspace. On creation it installs the OSMesa software-rendering
libraries, syncs the locked `web` and `dev` extras with `uv`, and fetches the
SO-101 assets. Codespaces machines have no GPU, so the container sets
`MUJOCO_GL=osmesa`.

Once the workspace is ready, start the headless host in its terminal:

```bash
uv run --extra web python scripts/run_sim.py --config configs/tasks/so101/pick_place.yaml --policy visual_servo --serve --seed 0
```

Port 8000 is declared as forwarded. Open it from the Ports panel and leave its
visibility `Private`, because the host has no authentication. If the asset
fetch hit the unauthenticated GitHub rate limit during creation, rerun
`uv run python scripts/fetch_assets.py --robot so101`.

Status: the headless host is covered by an acceptance test, but this container
configuration has not yet been run in a live Codespace.

## Browser Controls

The viewer supports orbit, pan, and zoom with the mouse. For the SO-101
keyboard mapping:

| Keys | Action |
| --- | --- |
| `W` / `S` | Cartesian X jog |
| `A` / `D` | Base yaw |
| `Q` / `E` | Cartesian Z jog |
| `R` / `F` | Gripper tilt/orientation |
| `C` | Close gripper while held |
| `O` | Open gripper while held |

The browser sends actions over WebSocket. It never calls MuJoCo directly. The
host validates each action against the registered robot capability contract
and applies it on the 30 Hz physics tick. When a policy is configured, manual
control temporarily owns the selected robot and the host returns to its hold
action or configured policy after the control lease expires.

The large scene viewport also includes an on-screen control pad. Its movement,
arm, and gripper keys mirror the keyboard controls, illuminate while held, and
can be pressed with a mouse or touchscreen. Keys unsupported by the active
robot are hidden.

## Appearance

The browser viewer follows the OS `prefers-color-scheme` by default (light or
dark) and can be overridden per browser with the header's theme button; the
choice persists in that browser's `localStorage`.

## Control Ownership, Rendering, And The API Surface

Covered in [docs/ARCHITECTURE.md](ARCHITECTURE.md#host--client-api): the
control-lease contract, the physics-thread/camera-worker threading model, and
the full REST/WebSocket route table. In short: reset and pause affect every
viewer; manual control uses a short per-instance lease; camera frames are
rendered by a decoupled worker thread and streamed over
`/api/camera/<name>/stream` (`multipart/x-mixed-replace`, consumed natively by
a plain `<img>` tag); the single-shot `/api/camera/<name>.jpg` snapshot
endpoint still exists for scripts and debugging.

### Camera panels

The sidebar's camera grid is a configurable list of panels, not a fixed
"front + wrist" pair: each panel has a dropdown of every camera name
`/api/robots` reports for the selected robot, "+ Camera" adds another panel,
and each panel has a remove button (at least one panel always stays). The
per-robot layout persists in that browser's `localStorage`, mirroring the
theme preference.

A policy may optionally publish extra named "debug" camera feeds alongside
the physical ones — for example, `visual_servo` (see
`research/classical_control/so101_visual_servo.py`) exposes
`front:detections` and `wrist:detections`, each an annotated copy of that
camera's frame with a crosshair at the last detected pixel, useful for
telling a detection failure apart from a control failure while tuning
visual servoing. These entries only appear once that policy is active, and
their panel stays on its last frame (retrying in the background) until the
first successful detection.

## Troubleshooting

Press `Ctrl+C` in the `run_sim.py` terminal to stop the desktop viewer, host,
and web server together. If the page stays on `connecting`, check the host
terminal and browser console. Hard-refresh after changing static viewer code
with `Ctrl+Shift+R`.

The browser client lives under `src/physai/web/static/`: `css/tokens.css`
(design tokens/theme) and `css/viewer.css` (layout/components), plus plain ES
modules under `js/` — `net.js` (WebSocket), `scene.js` (Three.js rendering),
`controls.js` (keyboard/control-pad jog), `ui.js` (status/telemetry/toast),
`cameras.js` (the multi-panel camera grid), and `main.js` (entry point).
There is no build step; edit a module and reload the page.

Three.js is loaded from a CDN, so the browser needs network access on the
first page load. Gamepad input, labels, and segmentation masks are not
implemented yet.
