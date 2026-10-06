# Web Viewer

The operating runbook: running the viewer, controls, recording and playback, cloud
workspace setup and troubleshooting. The host's API, threading model and control-lease
contract are in [ARCHITECTURE.md](ARCHITECTURE.md#host--client-api).

The viewer is a client of an already running simulation host. The host owns the robot, task
configuration, seed, policy, physics clock and command arbitration; the desktop viewer and
the browser render the same state and send control intents to that one host. Install the
training dependencies once (the browser viewer itself is in the base install), then use
plain `uv run`:

```bash
uv sync --extra training
```

## Start a simulation

With no `--policy`, the host holds its current pose and waits for browser jog commands. Add
`--viewer` for MuJoCo's desktop window on the same host:

```bash
MUJOCO_GL=egl uv run python scripts/run_sim.py --robot so101 --serve --seed 0
MUJOCO_GL=egl uv run python scripts/run_sim.py --manifest configs/manifests/so101_single_cube_fixed_place.yaml --policy scripted --serve --seed 0
MUJOCO_GL=egl uv run python scripts/run_sim.py --manifest configs/manifests/so101_single_cube_fixed_place.yaml --viewer --serve --host 0.0.0.0 --port 8004
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) or run
`uv run python scripts/run_web.py --connect http://127.0.0.1:8000`. `--manifest`, `--robot`,
`--seed` and `--policy` belong to the host; `run_web.py` only opens a client. The host reuses
`--seed` on each automatic reset, and a policy is reset only if one was supplied.

**Shared world.** Several heterogeneous robots in one MuJoCo scene, clock and model:

```bash
MUJOCO_GL=egl uv run python scripts/run_sim.py --manifest configs/manifests/heterogeneous_world.yaml --serve
```

Each instance has an ID, robot adapter, model path and world transform; it runs at 30 Hz
and starts idle. Selecting `arm_1` or `base_1` only changes which instance receives keyboard
commands; reset and pause affect the whole world.

**No desktop window.** On a server or container, pass only `--serve` (no `--viewer`). The host
runs until `Ctrl+C` or `SIGTERM` (handled the same, so a container stop is clean). Use
`MUJOCO_GL=osmesa` when there is no GPU or EGL device. The host has no authentication:
binding it to a non-loopback address lets anything that can reach the port send control
commands, so keep it on a private network or behind an authenticated tunnel.

**Isaac Sim.** `--sim isaac --serve` shows a mirror of the arm, table, target and cube
(`--viewer` and `--record-dir` stay MuJoCo-only,
[ADR 15](adr/simulators.md#adr-15-simulator-engine-selection)). The header shows which
simulator runs the session (a badge and the page title, from `GET /api/robots`).

## Cloud workspace (GitHub Codespaces)

`.devcontainer/devcontainer.json` describes a Python 3.12 container: on creation it installs
the OSMesa libraries, runs the locked `uv sync` and fetches the SO-101 assets, and it sets
`MUJOCO_GL=osmesa` (Codespaces has no GPU). Start the headless host in its terminal:

```bash
uv run python scripts/run_sim.py --manifest configs/manifests/so101_single_cube_fixed_place.yaml --policy visual_servo --serve --seed 0
```

Port 8000 is forwarded; keep its visibility `Private`. If the asset fetch hit GitHub's
unauthenticated rate limit, rerun `uv run python scripts/fetch_assets.py --robot so101`. The
headless host is covered by an acceptance test, but this container has not been run in a live
Codespace.

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
only, so it never moves the other three joints. Jogs ramp up the longer a key is held (a tap
is a small nudge, a hold ramps to a faster sweep over about 600 ms); the gripper steps at a
fixed rate. An on-screen control pad in the viewport mirrors the keys, can be pressed with a
mouse or touch, and hides keys the active robot does not support.

The browser sends actions over WebSocket and never calls MuJoCo. The host validates each
against the robot's capability contract and applies it on the 30 Hz tick. With a policy
configured, manual control owns the robot temporarily and the host returns to the hold action
or policy when the control lease expires.

**Tip pose.** Under the joint bars, `x`/`y`/`z` in mm and `roll`/`pitch`/`yaw` in degrees, in
the frame named beside the title (`base` for SO-101). Position is the pinch centre between the
fingertips, where an object is held (about 16 mm from the `gripperframe` site, within about 3
mm of a held cube's centre). Orientation is relative to a straight-down grasp: 0 / 0 / 0 means
the gripper points straight down with its jaws opening along world x (the scripted expert's
`top_down_quat`), intrinsic ZYX, and panning changes only yaw. The raw end-effector frame is
singular at a top-down grasp, hence this reference; past |pitch| = 80 degrees roll and yaw dim
with a "near gimbal lock" warning. A robot without a tool pose shows `ee_pose` in its own
frame, titled `end-effector`. The block is hidden in shared-world sessions.

**Gripper contact force.** For robots with a gripper, **Static pad** and **Moving pad** rows
show a dot that lights while the pad touches anything and the summed normal force in newtons
(MuJoCo's `mj_contactForce` for the latest step, friction excluded; table contact counts), or
`n/a` during playback because a restored pose cannot reproduce the squeeze. In the scripted
pick-and-place each pad reads about 3.8 N while holding the cube, and the net vertical force
equals its weight. Gripper torque is capped at 0.3 N·m in every mode
(`EnvConfig.gripper_force_limit`); the model's 3.35 N·m limit would drive about 34 N per pad and
sink the cube into the pads. The translucent orange boxes on the fingertips are the grasp pads,
the only collidable part of the fingers (`ManipulationSceneConfig` in
[common.py](../src/physai/sim/mujoco/scenes/common.py)); set the alpha of `pad_rgba` there to 0
to hide them. A held cube does not creep out because manipulation scenes run MuJoCo's no-slip
pass (`noslip_iterations = 5`).

### Camera panels

The sidebar's camera grid is a configurable list: each panel has a dropdown of every camera
`/api/robots` reports for the selected robot, "+ Camera" adds one, and each has a remove button
(at least one stays). The layout persists per robot in the browser's `localStorage`. Frames come
from a decoupled worker over `/api/camera/<name>/stream` (`multipart/x-mixed-replace`, consumed
by a plain `<img>`); `/api/camera/<name>.jpg` is a single snapshot for scripts and debugging.
A policy may publish extra "debug" feeds: `visual_servo` exposes `front:detections` and
`wrist:detections`, annotated copies of the camera frame with a crosshair at the last detected
pixel, which tell a detection failure from a control failure. They appear once the policy is
active and stay on their last frame until the first successful detection.

The theme follows `prefers-color-scheme` and can be overridden with the header button (stored
in `localStorage`).

## Recording episodes

Takes also carry `extras.*` arrays (joint velocity/effort, grip force, policy metrics; see `src/physai/data/extras.py`). Start the host with `--record-dir` to record from the browser (single robot only; refused with
a world, and requires `--serve`):

```bash
MUJOCO_GL=egl uv run python scripts/run_sim.py --robot so101 --serve --record-dir data/web_session
```

A Recording panel appears. **Record** starts a take; **Save ✓ success** or **Save ✗ fail** ends
it and writes `episode_XXXXX.npz` plus an updated `meta.json` with the success tag; **Discard**
writes nothing.
- The format is the one `scripts/collect_demos.py` writes
  ([Demonstration data](ARCHITECTURE.md#demonstration-data)) plus
  `observation.environment_state`. Each step stores the observation and the joint-target action
  the host applied, so jog input is recorded as resolved targets.
- Frames begin once every camera has produced an image (the panel says which it waits for). A
  world reset or a policy ending its episode discards the take.
- Pointing `--record-dir` at an existing dataset from the same robot continues its numbering; a
  dataset without `observation.environment_state` is rejected, so use a fresh directory.

## Episode playback

With `--record-dir`, a Playback panel lists saved episodes with their success tags. **Load**
pauses the world and shows frame 0. **|◀ / ▶|** (or the arrow keys) step one frame and stop
playing; the slider scrubs; **▶ / ❚❚** plays at 0.5x / 1x / 2x / 4x of the recorded real time
(it stops at the last frame; Play restarts it); **Exit** restores the world you interrupted,
still paused. Playback restores recorded simulator state (so objects are exact) and never
re-simulates; jog input, Reset, Resume and Record are refused while an episode is loaded, and
contact force reads `n/a`
([ADR 9](adr/web-host.md#adr-9-record-and-replay-web-sessions-through-the-existing-data-path)).
Episodes without `observation.environment_state` are listed but not playable.

To debug scripted demonstrations, collect with `--keep-failures` (failed episodes are discarded
by default) and open the dataset; the ✓/✗ tags come from the collection run. Do not press
**Record** in that session, since it would append your own take to the expert dataset. A dataset
from a different scene (for example `--sorting`, which has three cubes) has a different state
width and is refused.

```bash
uv run python scripts/collect_demos.py --episodes 5 --keep-failures --out data/debug_v1
MUJOCO_GL=egl uv run python scripts/run_sim.py --robot so101 --serve --record-dir data/debug_v1
```

## Troubleshooting

`Ctrl+C` in the `run_sim.py` terminal stops the desktop viewer, host and web server together.
If the page stays on `connecting`, check the host terminal and browser console; hard-refresh
(`Ctrl+Shift+R`) after changing static viewer code. The client is under
`src/physai/web/static/`: `css/tokens.css` and `css/viewer.css`, plain ES modules under `js/`
(`net.js`, `scene.js`, `controls.js`, `joints.js`, `ui.js`, `cameras.js`, `main.js`), no build
step. Three.js loads from a CDN, so the first page load needs network access. Gamepad input,
labels and segmentation masks are not implemented.

**Choppy viewer under WSL2.** MuJoCo can fall back to the CPU renderer (`llvmpipe`) even when
`nvidia-smi` sees the GPU. Enable the WSLg D3D12 renderer before opening the viewer (add the
`export` to `~/.bashrc` to persist it) and check with `glxinfo`; it should name `D3D12` and the
NVIDIA GPU, not `llvmpipe`. Do not install the Linux NVIDIA display driver inside WSL; it needs a
current driver on the Windows host and WSLg.

```bash
export GALLIUM_DRIVER=d3d12
uv run python scripts/run_sim.py --viewer
glxinfo -B | grep -Ei 'vendor|renderer|accelerated'
```

**Laggy camera feed on a Windows laptop with two GPUs.** The 3D view stays smooth (the browser
interpolates joint transforms) while camera panels lag, because a raw JPEG has no smoothing.
Windows picks the GPU per executable and does not know MuJoCo's offscreen renderer, so unless
the *exact* Python executable running the simulation is set to the discrete GPU it renders on
the integrated one. Check the renderer:

```bash
uv run python -c "
import ctypes
from mujoco.gl_context import GLContext
ctx = GLContext(320, 240); ctx.make_current()
glGetString = ctypes.windll.opengl32.glGetString; glGetString.restype = ctypes.c_char_p
print(glGetString(0x1F00), glGetString(0x1F01))
"
```

If it names the integrated GPU, target the real interpreter, not `sys.executable`: on a `uv` venv,
`.venv\Scripts\python.exe` is a launcher that starts the real interpreter as a child from uv's
cache, and only that child renders. With the host running, find it and set its preference (or use
Settings, System, Display, Graphics, Add an app, "High performance"):

```powershell
$py = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*run_sim.py*' -and $_.ExecutablePath -notlike '*\.venv\*' } | Select-Object -ExpandProperty ExecutablePath -Unique
New-ItemProperty -Path "HKCU:\Software\Microsoft\DirectX\UserGpuPreferences" -Name $py -Value "GpuPreference=2;" -PropertyType String -Force
```

It applies to new processes and is per machine. Even on the right GPU the stream is capped by
`CameraFeed.PERIOD` and `app.py`'s `_CAMERA_STREAM_PERIOD` (kept in sync, 30 fps), and by readback
and JPEG-encoding cost.
