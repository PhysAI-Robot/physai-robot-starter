# Web Viewer

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
MUJOCO_GL=egl uv run python scripts/run_sim.py \
  --robot so101 \
  --serve \
  --seed 0
```

To run the scripted pick-and-place policy instead, make it explicit:

```bash
MUJOCO_GL=egl uv run python scripts/run_sim.py \
  --config configs/tasks/so101/pick_place.yaml \
  --policy scripted \
  --serve \
  --seed 0
```

The desktop GUI and web server now use the same MuJoCo engine. Open
[http://127.0.0.1:8000/](http://127.0.0.1:8000/), or ask the client launcher to
open it:

```bash
uv run python scripts/run_web.py \
  --connect http://127.0.0.1:8000
```

Use another bind address or port on the host when needed:

```bash
MUJOCO_GL=egl uv run python scripts/run_sim.py \
  --config configs/tasks/so101/pick_place.yaml \
  --viewer --serve --host 0.0.0.0 --port 8004
```

The `--robot`, `--config`, `--seed`, and `--policy` options belong to the
simulation host. `run_web.py` does not select a robot or create a simulation.
For interactive viewer sessions, the host reuses the supplied `--seed` on each
automatic reset. A policy is reset only when one was explicitly supplied.

### Shared-world mode

To place multiple heterogeneous robot instances in one MuJoCo scene and clock,
use the world manifest:

```bash
MUJOCO_GL=egl uv run python scripts/run_sim.py \
  --world configs/worlds/heterogeneous.yaml \
  --serve
```

The manifest assigns each instance an ID, robot adapter, model path, and world
transform. It uses a 30 Hz control rate and starts idle. The browser renders
the complete scene. Selecting `arm_1` or `base_1` changes only which instance
receives keyboard commands; it does not start another simulation or remove the
other robot. Reset and pause affect the whole world.

## Run Without A Desktop Window

On a server, in a container, or in any environment with no display, replace
`--viewer` with `--headless`. The host and web server start exactly as above,
but no desktop window is opened:

```bash
MUJOCO_GL=egl uv run --extra web python scripts/run_sim.py \
  --config configs/tasks/so101/pick_place.yaml \
  --policy visual_servo \
  --headless \
  --serve \
  --seed 0
```

`--headless` requires `--serve` and cannot be combined with `--viewer`. The
host runs until `Ctrl+C` or `SIGTERM`, then stops the web server and the
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
uv run --extra web python scripts/run_sim.py \
  --config configs/tasks/so101/pick_place.yaml \
  --policy visual_servo \
  --headless \
  --serve \
  --seed 0
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

## Control Ownership

State is readable by every connected viewer, but manual control uses a short
lease. The first client sending a command becomes the active controller;
commands from another client are rejected while that lease is alive. Jogging
refreshes the lease, and releasing the connection or allowing the lease to
expire returns the host to a safe hold action, or resumes its configured
policy.

Reset and pause are host operations and affect every viewer. The desktop GUI
and browser therefore show the same step counter, transforms, camera frames,
pause state, and reset result.

## Rendering And Cameras

The scene manifest contains MuJoCo compiled meshes and authored material
colors. Dynamic geometry poses are streamed from MuJoCo; the browser does not
become a second physics engine. MuJoCo primitive cylinders use the viewer's
Z-up convention, while compiled robot meshes use the pose supplied by the
compiled geometry contract. The browser viewer presents that scene with a
horizontal studio floor, a neutral light background, and lighting tuned to
keep robot and task-object colors readable.

SO-101 `front` and `wrist` camera frames are rendered offscreen and cached by
the host, including in shared-world mode. In viewer/serve modes, a dedicated
camera worker copies the latest MuJoCo state and renders the feeds at about
5 FPS, so camera capture does not pause the 30 Hz physics loop. Camera
requests return the latest cached JPEG instead of stepping or rendering from
an HTTP worker. Available camera feeds are shown in the control panel so front
and wrist views remain legible.

## API Surface

- `/`: Three.js browser console
- `/api/robots`: registered robots and their action/camera capabilities
- `/api/scene`: static geometry manifest
- `/api/state`: latest transform snapshot
- `/api/mesh/<id>`: compiled mesh binary payload
- `/api/camera/<name>.jpg`: cached camera JPEG
- `/ws`: state stream and command channel

The WebSocket accepts `select_robot`, `command`, `reset`, and `pause` messages.
In shared-world mode, `/api/robots` reports instance IDs and their capability
contracts, while `/api/scene` and `/api/state` always describe the whole
world. The web client selects an existing instance and cannot create a robot.

## Troubleshooting

Press `Ctrl+C` in the `run_sim.py` terminal to stop the desktop viewer, host,
and web server together. If the page stays on `connecting`, check the host
terminal and browser console. Hard-refresh after changing static viewer code
with `Ctrl+Shift+R`.

Three.js is loaded from a CDN, so the browser needs network access on the first
page load. Gamepad input, detections, labels, masks, and annotation overlays
are not implemented yet.
