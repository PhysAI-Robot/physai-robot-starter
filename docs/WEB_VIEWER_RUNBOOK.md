# Web Viewer

The web console runs MuJoCo headlessly on the server and uses Three.js in the
browser as the primary viewer. Physics, control resolution, and camera capture
remain server-owned; the browser sends commands and renders the latest state.

## Start

Install the web extra and start one robot:

```bash
uv sync --extra web
MUJOCO_GL=egl uv run --extra web python scripts/run_web.py --robot so101
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/). The `MUJOCO_GL=egl`
setting selects the headless EGL renderer. Use another port when needed:

```bash
MUJOCO_GL=egl uv run --extra web python scripts/run_web.py \
  --robot so101 --port 8004
```

The scripted SO-101 demo remains available as a single-robot workflow:

```bash
MUJOCO_GL=egl uv run --extra web python scripts/run_web.py \
  --robot so101 --demo
```

`--demo` runs the SO-101 pick-and-place expert. It is intentionally not
available when starting a multi-robot fleet.

## Multi-Robot Fleet

Pass `--robot` more than once. Every selected robot gets its own simulation
session and physics loop, while the browser controls one selected robot at a
time:

```bash
MUJOCO_GL=egl uv run --extra web python scripts/run_web.py \
  --robot so101 \
  --robot turtlebot4 \
  --port 8004
```

Use the **Robot** selector in the page header to switch the active scene. The
selector reloads that robot's compiled meshes, state stream, camera feeds, and
capability-aware controls. Reset, pause, and keyboard commands target the
currently selected robot only. All sessions continue running in the server.

The web layer uses the shared robot registry and `RobotSpec`; it does not
contain SO-101 or TurtleBot4 physics logic. A new registered robot can join
the console without changing the viewer protocol. Robot-specific capabilities
determine which action modes and cameras are available.

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

The last gripper aperture is preserved while other keys are used. Controls are
sent over the WebSocket and resolved through the robot's declared action
contract. Robots without a gripper or Cartesian resolver should expose their
own capability-specific browser mapping before being driven from this UI.

## Rendering And Cameras

The scene manifest contains MuJoCo compiled meshes and authored material
colors. Dynamic geometry poses are streamed from MuJoCo; the browser does not
become a second physics engine. MuJoCo primitive cylinders use the viewer's
Z-up convention, while compiled robot meshes use the pose supplied by the
compiled geometry contract.

SO-101 front and wrist images are rendered offscreen and cached by the session.
Web camera requests return the latest cached JPEG instead of rendering from an
HTTP worker, so camera polling does not block the physics command path.

## API Surface

- `/`: Three.js browser console
- `/api/robots`: registered robots and their action/camera capabilities
- `/api/scene?robot=<name>`: static geometry manifest for one robot
- `/api/state?robot=<name>`: latest transform snapshot
- `/api/mesh/<id>?robot=<name>`: compiled mesh binary payload
- `/api/camera/<name>.jpg?robot=<name>`: cached camera JPEG
- `/ws`: state stream and command channel

The WebSocket accepts `select_robot`, `command`, `reset`, and `pause` messages.
Commands may include a `robot` field; otherwise they target the connection's
currently selected robot.

## Troubleshooting

Press `Ctrl+C` in the server terminal to stop Uvicorn and close all sessions.
If the page stays on `connecting`, check the server log and browser console.
Hard-refresh after changing static viewer code with `Ctrl+Shift+R`.

Three.js is loaded from a CDN, so the browser needs network access on the first
page load. Gamepad input, detections, labels, masks, and annotation overlays
are not implemented yet. The viewer currently supports keyboard controls for
robots whose capabilities match the supported jog contract.
