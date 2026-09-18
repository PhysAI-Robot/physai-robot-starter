# Web Viewer

The web viewer is a client of an already running simulation host. The host
owns the robot, task configuration, seed, policy, MuJoCo model, physics clock,
and command arbitration. The desktop viewer and browser render the same state
and send control intents to that one host.

## Start One Authoritative Simulation

Start the configured simulation with the desktop viewer and web server:

```bash
MUJOCO_GL=egl uv run --extra web python scripts/run_sim.py \
  --config configs/tasks/so101/pick_place.yaml \
  --viewer \
  --serve \
  --seed 0
```

The desktop GUI and web server now use the same MuJoCo engine. Open
[http://127.0.0.1:8000/](http://127.0.0.1:8000/), or ask the client launcher to
open it:

```bash
uv run --extra web python scripts/run_web.py \
  --connect http://127.0.0.1:8000
```

Use another bind address or port on the host when needed:

```bash
MUJOCO_GL=egl uv run --extra web python scripts/run_sim.py \
  --config configs/tasks/so101/pick_place.yaml \
  --viewer --serve --host 0.0.0.0 --port 8004
```

The `--robot`, `--config`, `--seed`, and `--policy` options belong to the
simulation host. `run_web.py` does not select a robot or create a simulation.
For interactive viewer sessions, the host reuses the supplied `--seed` on each
automatic reset so the scripted task returns to the same reproducible scene.

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
and applies it on the physics tick.

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
compiled geometry contract.

SO-101 camera frames are rendered offscreen and cached by the host. Camera
requests return the latest cached JPEG instead of stepping or rendering from
an HTTP worker.

## API Surface

- `/`: Three.js browser console
- `/api/robots`: registered robots and their action/camera capabilities
- `/api/scene`: static geometry manifest
- `/api/state`: latest transform snapshot
- `/api/mesh/<id>`: compiled mesh binary payload
- `/api/camera/<name>.jpg`: cached camera JPEG
- `/ws`: state stream and command channel

The WebSocket accepts `select_robot`, `command`, `reset`, and `pause` messages.
The current host runs one authoritative robot session; the robot identity is
reported by `/api/robots` and is not created by the web client.

## Troubleshooting

Press `Ctrl+C` in the `run_sim.py` terminal to stop the desktop viewer, host,
and web server together. If the page stays on `connecting`, check the host
terminal and browser console. Hard-refresh after changing static viewer code
with `Ctrl+Shift+R`.

Three.js is loaded from a CDN, so the browser needs network access on the first
page load. Gamepad input, detections, labels, masks, and annotation overlays
are not implemented yet.
