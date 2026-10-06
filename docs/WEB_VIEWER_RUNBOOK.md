# Web Viewer

The operating runbook: running the viewer, controls, recording and playback, cloud
workspace setup and troubleshooting. The host's API, threading model and control-lease
contract are in [ARCHITECTURE.md](ARCHITECTURE.md#host-and-client-api).

The viewer is a client of an already running simulation host. The host owns the robot, task
configuration, seed, policy, physics clock and command arbitration; the desktop viewer and
the browser render the same state and send control intents to that one host. The browser
viewer is in the base install; `uv sync --extra training` adds the policies that need torch.

## Start a simulation

With no `--policy`, the host holds its current pose and waits for browser jog commands. Add
`--viewer` for MuJoCo's desktop window on the same host:

```bash
MUJOCO_GL=egl uv run python scripts/run_sim.py --serve --seed 0
MUJOCO_GL=egl uv run python scripts/run_sim.py --manifest configs/manifests/so101_single_cube_fixed_place.yaml --policy scripted --serve --seed 0
MUJOCO_GL=egl uv run python scripts/run_sim.py --manifest configs/manifests/so101_single_cube_fixed_place.yaml --viewer --serve --host 0.0.0.0 --port 8004
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) or run
`uv run python scripts/run_web.py --connect http://127.0.0.1:8000`. `--manifest`, `--seed` and
`--policy` belong to the host; `run_web.py` only opens a client. The host reuses `--seed` on
each automatic reset.

**Shared world.** Several heterogeneous robots in one MuJoCo scene, clock and model:

```bash
MUJOCO_GL=egl uv run python scripts/run_sim.py --manifest configs/manifests/heterogeneous_world.yaml --serve
```

Each instance has an ID, robot adapter, model path and world transform, runs at 30 Hz and
starts idle. Selecting an instance only changes which one receives keyboard commands; reset
and pause affect the whole world.

**No desktop window.** On a server or container, pass only `--serve`. The host runs until
`Ctrl+C` or `SIGTERM`. Use `MUJOCO_GL=osmesa` when there is no GPU or EGL device. The host has
no authentication: binding it to a non-loopback address lets anything that can reach the port
send control commands, so keep it on a private network or behind an authenticated tunnel.

**Isaac Sim.** `--sim isaac --serve` shows a mirror of the arm, table, target and cube
(`--viewer` and `--dataset` stay MuJoCo-only,
[DECISIONS.md D](DECISIONS.md#d-isaac-sim-is-an-optional-peer-engine)). The header badge shows
which simulator runs the session.

**GitHub Codespaces.** `.devcontainer/devcontainer.json` installs the OSMesa libraries, runs
the locked `uv sync`, fetches the SO-101 assets and sets `MUJOCO_GL=osmesa`; port 8000 is
forwarded (keep it `Private`). Start the host with `--policy visual_servo --serve`. This
container has not been run in a live Codespace; the headless host is covered by an acceptance
test.

## Browser controls

Mouse: orbit, pan, zoom. SO-101 keyboard mapping:

| Keys | Action |
| --- | --- |
| `W` / `S` | Cartesian X jog |
| `Q` / `E` | Cartesian Z jog |
| `A` / `D` | `shoulder_pan`, direct joint jog |
| `I` / `K` | `wrist_flex`, direct joint jog |
| `J` / `L` | `wrist_roll`, direct joint jog |
| `R` / `F` | Open / close gripper while held |

The X/Z jog targets a point at the wrist and is solved over `shoulder_lift` and `elbow_flex`
only. Jogs ramp up the longer a key is held; an on-screen control pad mirrors the keys and
hides keys the active robot does not support. The browser sends actions over WebSocket and
never calls MuJoCo; the host validates each against the robot's capability contract and
applies it on the 30 Hz tick. With a policy configured, manual control owns the robot
temporarily and the host returns to the hold action or policy when the control lease expires.

**Tip pose.** Under the joint bars: `x`/`y`/`z` in mm and `roll`/`pitch`/`yaw` in degrees in
the frame named beside the title (`base` for SO-101). Position is the pinch centre between the
fingertips; orientation is relative to a straight-down grasp (0 / 0 / 0, intrinsic ZYX), with a
"near gimbal lock" warning past |pitch| = 80 degrees. Hidden in shared-world sessions.

**Gripper contact force.** For robots with a gripper, **Static pad** and **Moving pad** rows
show whether the pad touches anything and the summed normal force in newtons (`n/a` during
playback). Gripper torque is capped at 0.3 N·m (`EnvConfig.gripper_force_limit`), and the
translucent orange boxes on the fingertips are the grasp pads, the only collidable part of the
fingers (`ManipulationSceneConfig.pad_rgba`).

**Camera panels.** The sidebar's camera grid is a configurable list: each panel has a dropdown
of every camera `/api/robots` reports for the selected robot, "+ Camera" adds one, and the
layout persists per robot in `localStorage`. Frames stream from
`/api/camera/<name>/stream`; `/api/camera/<name>.jpg` is a snapshot for scripts. A policy may
publish extra debug feeds (`visual_servo`: `front:detections`, `wrist:detections`, the camera
frame with a crosshair at the last detection). The theme follows `prefers-color-scheme` and
can be overridden with the header button.

## Recording and playback

Start the host with `--dataset` to record from the browser (single robot only, requires
`--serve`):

```bash
MUJOCO_GL=egl uv run python scripts/run_sim.py --serve --dataset data/web_session
```

A Recording panel appears. **Record** starts a take; **Save ✓ success** or **Save ✗ fail** ends
it and writes `episode_XXXXX.npz` plus an updated `meta.json` with the success tag;
**Discard** writes nothing. The format is the one `collect_demos.py` writes
([Demonstration data](ARCHITECTURE.md#demonstration-data)) plus `observation.environment_state`
and `extras.*` arrays (`src/physai/data/extras.py`); jog input is recorded as resolved joint
targets. Frames begin once every camera has produced an image, and a world reset or a policy
ending its episode discards the take. `--dataset` pointed at an existing dataset continues its
numbering; one without `observation.environment_state` is rejected, so use a fresh directory.

With `--dataset`, a Playback panel lists saved episodes. **Load** pauses the world and shows
frame 0; the arrow keys step one frame, the slider scrubs, **▶ / ❚❚** plays at 0.5x to 4x of
the recorded real time, and **Exit** restores the world you interrupted, still paused. Jog,
Reset, Resume and Record are refused while an episode is loaded. Episodes without
`observation.environment_state` are listed but not playable, and a dataset from a different
scene (for example `--sorting`) has a different state width and is refused.

To debug scripted demonstrations, collect with `--keep-failures` and open the dataset (do not
press **Record** in that session, it would append your own take):

```bash
uv run python scripts/collect_demos.py --episodes 5 --keep-failures --dataset data/debug_v1
MUJOCO_GL=egl uv run python scripts/run_sim.py --serve --dataset data/debug_v1
```

## Troubleshooting

`Ctrl+C` in the `run_sim.py` terminal stops the desktop viewer, host and web server together.
If the page stays on `connecting`, check the host terminal and browser console, and hard-refresh
(`Ctrl+Shift+R`) after changing static code. The client is under `src/physai/web/static/`
(`css/` and plain ES modules under `js/`, no build step). Three.js loads from a CDN, so the
first page load needs network access.

**Choppy viewer under WSL2.** MuJoCo can fall back to the CPU renderer (`llvmpipe`). Run
`export GALLIUM_DRIVER=d3d12` before `--viewer` and check `glxinfo -B`: it should name `D3D12`
and the NVIDIA GPU. It needs a current Windows driver and WSLg; do not install the Linux
NVIDIA display driver inside WSL.

**Laggy camera feed on a Windows laptop with two GPUs.** Windows picks the GPU per executable,
so unless the exact Python interpreter running the simulation is set to the discrete GPU
(Settings, System, Display, Graphics, Add an app, "High performance") it renders on the
integrated one. On a `uv` venv, `.venv\Scripts\python.exe` is only a launcher: set the
preference on the real interpreter that runs as its child (the process whose command line
contains `run_sim.py`). The stream is also capped at 30 fps by `CameraFeed.PERIOD` and
`app.py`'s `_CAMERA_STREAM_PERIOD`, which must stay in sync.
