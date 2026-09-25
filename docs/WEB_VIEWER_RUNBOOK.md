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

Under the joint bars, a **Tip pose** block shows where the gripper tip is:
`x`/`y`/`z` in millimetres and `roll`/`pitch`/`yaw` in degrees, in the frame
named beside the title (`base` for the SO-101).

- **Position** is the pinch centre, the point between the fingertips where an
  object is held. It sits about 16 mm from the `gripperframe` site; while the
  scripted grasp holds the cube it is within ~3 mm of the cube's centre,
  versus ~15 mm for the site.
- **Orientation** is relative to a straight-down grasp, so **0° / 0° / 0°
  means the gripper points straight down** with its jaws opening along the
  world x axis (the reference the scripted expert uses, `top_down_quat`).
  Roll/pitch/yaw are intrinsic ZYX angles computed in the browser from the
  reported quaternion. Panning the arm changes only yaw; a grasp reads about
  0° roll and pitch with the yaw of the approach. During a full scripted
  pick-and-place, pitch stays between -36° and 11°, and while the cube is
  held roll and pitch stay within a few degrees of 0.
- **Gimbal lock:** the SO-101's raw end-effector frame is singular exactly at
  a top-down grasp (its approach axis is x, so ZYX pitch sits at ±90° and roll
  and yaw swing wildly), which is why the reference is used. The singular
  pose is now a horizontal approach with the jaws stacked vertically, which
  table-top grasping does not use. Past |pitch| = 80° roll and yaw are dimmed
  and the block shows a "near gimbal lock" warning.
- A robot without a tool pose reports its observation's `ee_pose` in its own
  frame instead, and the title says `end-effector` rather than `tip`.

The block is hidden in `--world` sessions and for robots that report no pose.

## Gripper Contact Force

For robots with a gripper, the joint panel lists a **Static pad** and a
**Moving pad** row. The dot lights while that pad touches anything, and the
number beside it is the normal force in newtons, summed over everything the
pad is touching (`–` when it touches nothing). The value is MuJoCo's contact
normal force from the latest physics step (`mj_contactForce`); friction is
not included. Contact with the table counts the same as contact with an
object.

Sanity reference from the scripted pick-and-place: the two pads read the same
force while holding the cube (about 3.8 N each), and their net vertical force
equals the cube's weight. The gripper torque is capped at 0.3 N·m in every
mode. The cap matters: the model's own limit is 3.35 N·m, which drives about
34 N per pad against a 0.29 N cube and pushes the cube roughly 10 mm into the
pads and fingers, and below roughly 0.1 N·m (about 1 N per pad) the scripted
grasp drops the cube.

Holding the gripper key (`G`) on a lifted cube used to let it slide slowly
out of the fingers and fall after ~15 s, at any grip force. That is friction
creep, not slip: the contact reported only 0.13-0.19 N of friction against
a limit of about 8 N. MuJoCo softens friction, so a constant load (the cube's
weight) moves the contact at ~1 mm/s. Manipulation scenes now run the
solver's no-slip pass (`noslip_iterations = 5` in `ManipulationSceneConfig`),
after which a held cube moves 0.0 mm over 30 s, for ~30% more solver time.

The translucent orange boxes on the fingertips are those pads: the only parts
of the fingers that collide. They are fitted to the flat face at each fingertip
(the last ~6 mm; behind it the lattice is recessed): flush with it, along its
angle, and one cube width apart at their centres when the gripper is at
0.16 rad. The moving finger's face is tilted about 8 degrees from the static
one (the fingers form a V), and its pad is tilted to match. To hide them, set the alpha of
`pad_rgba` in `ManipulationSceneConfig`
([common.py](../src/physai/sim/scenes/common.py)) to 0.

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

The 3D viewport can look smooth while the camera panels feel choppy even
though the machine has a capable discrete GPU. The 3D view stays smooth
regardless, because the browser interpolates received joint/geometry
transforms every animation frame; a raw camera JPEG has no such
interpolation, so any rendering slowdown shows up directly as choppiness.

On a laptop with both an integrated and a discrete GPU, Windows decides
per-executable which GPU handles rendering, and it does not know about
MuJoCo's offscreen renderer. Unless the *exact* Python executable that runs
the simulation is explicitly pointed at the discrete GPU, Windows silently
defaults it to the integrated one, which is far slower at the
`renderer.render()` calls `Host._camera_loop` does every capture (see
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
Graphics`) instead of the discrete one, find the exact interpreter path to
target. Don't trust `sys.executable` for this: on a `uv`-managed venv,
`.venv\Scripts\python.exe` is a small launcher (tens of KB, not a full
interpreter) that reports itself as `sys.executable` for compatibility, but
when running a script file it re-launches the real, long-running interpreter
as a *child process* from uv's own cache — and it's that child's path
Windows' GPU preference actually keys off, not the launcher's. Confirmed by
inspecting the live process tree while `--serve` is running: `uv.exe` →
`.venv\Scripts\python.exe` → `%APPDATA%\uv\python\<version>\python.exe`, with
only the last one doing any rendering work. With the host already running,
find that real path directly:

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

This takes effect immediately for new processes; no reboot or `uv` reinstall
needed. Re-run the `GL_RENDERER` check above to confirm. This is a per-machine
Windows setting, not a repository file, so it does not travel with the repo
and must be set again on any other Windows machine that hits the same
symptom.

Even with the correct GPU in use, the camera stream is still capped by
`Host._CAMERA_PERIOD`/`app.py`'s `_CAMERA_STREAM_PERIOD` (kept in sync) — the
default targets 30 fps, but the actually achieved rate depends on readback
and JPEG-encoding overhead, not just the GPU.

Three.js is loaded from a CDN, so the browser needs network access on the
first page load. Gamepad input, labels, and segmentation masks are not
implemented yet.
