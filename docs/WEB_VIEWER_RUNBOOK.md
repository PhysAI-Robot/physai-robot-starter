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
MUJOCO_GL=egl uv run python scripts/run_sim.py --manifest configs/manifests/so101_pick_place.yaml --policy scripted --serve --seed 0
```

The desktop GUI and web server now use the same MuJoCo engine. Open
[http://127.0.0.1:8000/](http://127.0.0.1:8000/), or ask the client launcher to
open it:

```bash
uv run python scripts/run_web.py --connect http://127.0.0.1:8000
```

Use another bind address or port on the host when needed:

```bash
MUJOCO_GL=egl uv run python scripts/run_sim.py --manifest configs/manifests/so101_pick_place.yaml --viewer --serve --host 0.0.0.0 --port 8004
```

The `--manifest`, `--robot`, `--seed`, and `--policy` options belong to the
simulation host. `run_web.py` does not select a robot or create a simulation.
For interactive viewer sessions, the host reuses the supplied `--seed` on each
automatic reset. A policy is reset only when one was explicitly supplied.

### Shared-world mode

To place multiple heterogeneous robot instances in one MuJoCo scene and clock,
use the world manifest:

```bash
MUJOCO_GL=egl uv run python scripts/run_sim.py --manifest configs/manifests/heterogeneous_world.yaml --serve
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
MUJOCO_GL=egl uv run --extra web python scripts/run_sim.py --manifest configs/manifests/so101_pick_place.yaml --policy visual_servo --serve --seed 0
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
uv run --extra web python scripts/run_sim.py --manifest configs/manifests/so101_pick_place.yaml --policy visual_servo --serve --seed 0
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
| `Q` / `E` | Cartesian Z jog |
| `A` / `D` | Shoulder pan (`shoulder_pan`), direct joint jog |
| `I` / `K` | Wrist tilt (`wrist_flex`), direct joint jog |
| `J` / `L` | Wrist roll (`wrist_roll`), direct joint jog |
| `R` / `F` | Open/close gripper while held |

The X/Z jog (`W`/`S`/`Q`/`E`) targets a point at the wrist rather than the
gripper tip, and is solved only over `shoulder_lift`/`elbow_flex`, so it
never moves `shoulder_pan`, `wrist_flex`, or `wrist_roll`; those three are
jogged directly with `A`/`D`, `I`/`K`, and `J`/`L` instead. All of these ramp
up the longer a key is held — a quick tap gives a small, precise nudge, and a
sustained hold ramps up to a faster sweep over about 600ms. The gripper
(`R`/`F`) does not ramp; it steps at a fixed rate while held.

The browser sends actions over WebSocket. It never calls MuJoCo directly. The
host validates each action against the registered robot capability contract
and applies it on the 30 Hz physics tick. When a policy is configured, manual
control temporarily owns the selected robot and the host returns to its hold
action or configured policy after the control lease expires.

The large scene viewport also includes an on-screen control pad. Its movement,
arm, and gripper keys mirror the keyboard controls, illuminate while held, and
can be pressed with a mouse or touchscreen. Keys unsupported by the active
robot are hidden.

## Tip Pose Readout

Under the joint bars, a **Tip pose** block shows the gripper tip: `x`/`y`/`z`
in millimetres and `roll`/`pitch`/`yaw` in degrees, in the frame named beside
the title (`base` for the SO-101).

- **Position** is the pinch centre between the fingertips, where an object is
  held (about 16 mm from the `gripperframe` site, and within ~3 mm of a held
  cube's centre).
- **Orientation** is relative to a straight-down grasp: **0° / 0° / 0° means
  the gripper points straight down** with its jaws opening along the world x
  axis (the scripted expert's `top_down_quat`). Angles are intrinsic ZYX,
  computed in the browser from the reported quaternion; panning the arm
  changes only yaw.
- **Gimbal lock:** the SO-101's raw end-effector frame is singular at a
  top-down grasp, which is why this reference is used. The singular pose is a
  horizontal approach that table-top grasping does not use; past
  |pitch| = 80° roll and yaw are dimmed with a "near gimbal lock" warning.
- A robot without a tool pose shows its observation's `ee_pose` in its own
  frame, titled `end-effector` instead of `tip`.

The block is hidden in `--world` sessions and for robots that report no pose.

## Gripper Contact Force

For robots with a gripper, the joint panel lists a **Static pad** and a
**Moving pad** row. The dot lights while that pad touches anything, and the
number is the summed normal force in newtons over everything the pad touches
(`–` when nothing; table contact counts like object contact). It is MuJoCo's
`mj_contactForce` from the latest step, friction excluded, and it reads `–`
during playback because a restored pose cannot reproduce the recorded squeeze.

Reference from the scripted pick-and-place: each pad reads about 3.8 N while
holding the cube, and the net vertical force equals the cube's weight. Gripper
torque is capped at 0.3 N·m in every mode (`EnvConfig.gripper_force_limit`);
the model's own 3.35 N·m limit would drive about 34 N per pad and sink the
cube into the pads.

The translucent orange boxes on the fingertips are the grasp pads, the only
parts of the fingers that collide, fitted to each fingertip's flat face
(`ManipulationSceneConfig` in
[common.py](../src/physai/sim/scenes/common.py)); set the alpha of `pad_rgba`
there to 0 to hide them. A held cube does not creep out of the fingers because
manipulation scenes run MuJoCo's no-slip pass (`noslip_iterations = 5`).

## Recording Episodes

Start the host with `--record-dir` to record from the browser into a dataset
directory:

```bash
MUJOCO_GL=egl uv run python scripts/run_sim.py --robot so101 --serve --record-dir data/web_session
```

A Recording panel appears in the sidebar. **Record** starts a take; then
**Save ✓ success** or **Save ✗ fail** ends it and writes one
`episode_XXXXX.npz` plus an updated `meta.json` with that success tag, and
**Discard** drops it without writing anything. The panel shows saved episodes,
successes, and the frame count of the take in progress.

- The format is the same one `scripts/collect_demos.py` writes (see
  [Demonstration data](ARCHITECTURE.md#demonstration-data)), plus an
  `observation.environment_state` key holding the full simulator state.
- Each step stores the observation and the joint-target action the host
  actually applied, so jog input is recorded as resolved joint targets.
- Frames begin once every camera has produced an image; the panel says which
  camera it is waiting for.
- A world reset (or a policy ending its episode) discards the take in
  progress, and the panel reports it.
- Pointing `--record-dir` at an existing dataset from the same robot continues
  its numbering. A dataset without `observation.environment_state` (one
  collected before `collect_demos.py` began saving it) is rejected; use a
  fresh directory.
- Recording is single-robot only: `--record-dir` is refused with `--world`, and
  requires `--serve`.

## Episode Playback

With `--record-dir` set, a Playback panel lists the saved episodes with their
success tags. Pick one and press **Load**: the world pauses and shows frame 0
of that episode. Then:

- **◀ / ▶** (or the arrow keys) step one frame and stop playing; the slider
  scrubs to any frame.
- **Play** runs the episode at **0.5× / 1× / 2× / 4×**. Speed follows the wall
  clock, so 1× is the real time the episode was recorded at (its control
  rate), and playback stops at the last frame; Play again restarts it.
- **Exit** puts back exactly the world you interrupted, still paused; press
  Resume to carry on live.

It replays recorded simulator state (`observation.environment_state`, so the
cube and other objects are exact) on the same paused world and never
re-simulates. While an episode is loaded, jog input, Reset, Resume and Record
are refused, and each contact-force readout shows `n/a` (see
[ADR 9](adr/0009-web-session-recording-and-playback.md)). The tip pose is
recomputed from the restored state. Episodes without
`observation.environment_state` (datasets collected before `collect_demos.py`
began saving it) are listed but not playable.

### Debugging scripted demonstrations

`scripts/collect_demos.py` saves the same state, so you can replay what the
scripted expert did. Failed episodes are discarded by default, so collect
with `--keep-failures` to inspect them, then open the dataset:

```bash
uv run python scripts/collect_demos.py --episodes 5 --keep-failures --out data/debug_v1
MUJOCO_GL=egl uv run python scripts/run_sim.py --robot so101 --serve --record-dir data/debug_v1
```

Load an episode and step through it; the ✓/✗ tags come from the collection
run. Do not press **Record** in this session: it would append your own take to
the expert dataset. Viewing alone writes nothing. A dataset from a different
scene (for example `--sorting`, which has three cubes) has a different state
width and is refused with a clear error.

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
`controls.js` (keyboard/control-pad jog), `joints.js` (the joint-state and
gripper-contact HUD), `ui.js` (status/telemetry/toast), `cameras.js` (the
multi-panel camera grid), and `main.js` (entry point). There is no build
step; edit a module and reload the page.

### Choppy viewer under WSL2

Only for users running the viewer inside WSL2; native Ubuntu can skip this. On
WSL2, MuJoCo can fall back to the CPU software renderer (`llvmpipe`), which
makes the interactive viewer look choppy even when `nvidia-smi` sees the
NVIDIA GPU. Enable the WSLg D3D12 renderer for the shell before opening the
viewer:

```bash
export GALLIUM_DRIVER=d3d12
uv run python scripts/run_sim.py --viewer
```

To apply it to future Bash sessions, add it once:

```bash
printf '\nexport GALLIUM_DRIVER=d3d12\n' >> ~/.bashrc
source ~/.bashrc
```

Verify that OpenGL is accelerated and reports the NVIDIA GPU:

```bash
glxinfo -B | grep -Ei 'vendor|renderer|accelerated'
```

The renderer should mention `D3D12` and the NVIDIA GPU, not `llvmpipe` or
`Accelerated: no`. WSL2 GPU support needs a current NVIDIA driver on the
Windows host and WSLg; do not install the Linux NVIDIA display driver inside
WSL with `sudo apt install nvidia-driver`.

### Laggy camera feed on a Windows laptop with two GPUs

The 3D viewport can look smooth while the camera panels feel choppy: the
browser interpolates joint transforms every animation frame, but a raw camera
JPEG has no such smoothing, so slow rendering shows up directly.

On a laptop with an integrated and a discrete GPU, Windows picks the GPU per
executable and does not know about MuJoCo's offscreen renderer. Unless the
*exact* Python executable running the simulation is pointed at the discrete
GPU, it defaults to the integrated one, which is far slower at the
`renderer.render()` calls the `CameraFeed` worker makes (see
[docs/ARCHITECTURE.md](ARCHITECTURE.md#host--client-api)).

Check which GPU is actually rendering:

```bash
uv run python -c "
import ctypes
from mujoco.gl_context import GLContext

ctx = GLContext(320, 240)
ctx.make_current()
opengl32 = ctypes.windll.opengl32
glGetString = opengl32.glGetString
glGetString.restype = ctypes.c_char_p
print('GL_VENDOR:  ', glGetString(0x1F00))
print('GL_RENDERER:', glGetString(0x1F01))
"
```

If `GL_RENDERER` names the integrated GPU (for example `Intel(R) UHD
Graphics`), find the exact interpreter path to target. Don't use
`sys.executable`: on a `uv` venv, `.venv\Scripts\python.exe` is a small
launcher that re-launches the real interpreter as a *child process* from uv's
cache (`uv.exe` → `.venv\Scripts\python.exe` →
`%APPDATA%\uv\python\<version>\python.exe`), and only that child renders, so
its path is what Windows' GPU preference keys off. With the host running, find
it directly:

```powershell
Get-CimInstance Win32_Process | Where-Object {
  $_.Name -eq 'python.exe' -and $_.CommandLine -like '*run_sim.py*' -and $_.ExecutablePath -notlike '*\.venv\*'
} | Select-Object -ExpandProperty ExecutablePath -Unique
```

Then point that exact path at the discrete GPU, either through Windows
Settings (Settings → System → Display → Graphics → Add an app → browse to
that path → set it to "High performance"), or from PowerShell:

```powershell
$py = "<paste the path found above>"
New-ItemProperty -Path "HKCU:\Software\Microsoft\DirectX\UserGpuPreferences" `
  -Name $py -Value "GpuPreference=2;" -PropertyType String -Force
```

This takes effect immediately for new processes; re-run the `GL_RENDERER`
check to confirm. It is a per-machine Windows setting, so it must be set again
on any other machine with the same symptom.

Even on the right GPU, the camera stream is capped by
`CameraFeed.PERIOD`/`app.py`'s `_CAMERA_STREAM_PERIOD` (kept in sync, 30 fps
by default), and the achieved rate also depends on readback and JPEG-encoding
overhead.

Three.js is loaded from a CDN, so the browser needs network access on the
first page load. Gamepad input, labels, and segmentation masks are not
implemented yet.
