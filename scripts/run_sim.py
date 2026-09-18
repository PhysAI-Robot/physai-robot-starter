"""Run one episode, optionally writing a video. The 30-second sanity check.

python scripts/run_sim.py                      # scripted expert, 1 episode
python scripts/run_sim.py --config configs/tasks/so101/pick_place.yaml
python scripts/run_sim.py --episodes 5 --seed 0
python scripts/run_sim.py --video --episodes 5 --seed 0
python scripts/run_sim.py --policy constant    # baseline: do nothing
python scripts/run_sim.py --policy lerobot --checkpoint outputs/act_ckpt
python scripts/run_sim.py --viewer             # single-window scene + cameras UI
python scripts/run_sim.py --viewer --serve     # GUI plus shared web host
"""

from __future__ import annotations

import argparse
import tempfile
import threading
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401
import mujoco
import numpy as np

from physai.config import (
    DomainRandomizationConfig,
    TaskConfig,
    load_sim_config,
    load_task_config,
)
from physai.policy import available_policies, create_policy
from physai.robots import available_robots, create_robot
from physai.robots.so101 import EnvConfig
from physai.robots.turtlebot import TurtleBot4Config
from physai.sim import SceneConfig
from physai.tasks import TaskRuntime, create_task
from physai.web.runtime import SimulationHost


def write_video(frames: np.ndarray, stem: Path, fps: int) -> Path:
    """Write mp4 if an H.264 encoder is available, otherwise fall back to GIF.

    imageio's default pyav path raises an unhelpful `expected bytes, NoneType`
    when no codec is registered, so the codec is named explicitly and the
    fallback is silent-but-reported rather than a stack trace.
    """
    import imageio.v3 as iio

    mp4 = stem.with_suffix(".mp4")
    for plugin, kwargs in (
        ("FFMPEG", {"codec": "libx264"}),
        ("pyav", {"codec": "libx264"}),
    ):
        try:
            iio.imwrite(mp4, frames, fps=fps, plugin=plugin, **kwargs)
            return mp4
        except (ImportError, OSError, RuntimeError, TypeError, ValueError):
            continue

    gif = stem.with_suffix(".gif")
    iio.imwrite(gif, frames[::2], duration=2000 / fps, loop=0)
    print("  (no H.264 encoder found — wrote a GIF; run `uv sync` for mp4 support)")
    return gif


class SingleWindowViewer:
    """Tk client rendering an authoritative host and its model cameras."""

    def __init__(self, host: SimulationHost, camera_names: list[str]) -> None:
        try:
            import tkinter as tk
        except ImportError as exc:
            raise RuntimeError(
                "the custom viewer requires Tkinter; install python3-tk"
            ) from exc
        self._tk = tk
        self.root = tk.Tk()
        self.root.title("PhysAI MuJoCo Viewer")
        self.host = host
        self.camera_names = camera_names
        self.closed = False
        self.paused = False
        self._last_drag: tuple[int, int] | None = None
        scene_width = min(800, int(host.model.vis.global_.offwidth))
        scene_height = min(600, int(host.model.vis.global_.offheight))
        self._scene_renderer = mujoco.Renderer(
            host.model, height=scene_height, width=scene_width
        )
        self._free_camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(host.model, self._free_camera)

        toolbar = tk.Frame(self.root)
        toolbar.pack(fill=tk.X)
        self.pause_button = tk.Button(toolbar, text="Pause", command=self.toggle_pause)
        self.pause_button.pack(side=tk.LEFT, padx=4, pady=4)
        tk.Button(toolbar, text="Reset", command=self.reset).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="Zoom +", command=lambda: self.zoom(0.85)).pack(
            side=tk.LEFT, padx=4
        )
        tk.Button(toolbar, text="Zoom -", command=lambda: self.zoom(1.18)).pack(
            side=tk.LEFT, padx=4
        )
        self.status = tk.Label(toolbar, text="running", anchor="w")
        self.status.pack(side=tk.LEFT, padx=12)

        content = tk.Frame(self.root)
        content.pack(fill=tk.BOTH, expand=True)
        self.scene_label = tk.Label(content, text="Rendering scene...", bg="#202124")
        self.scene_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scene_label.bind("<ButtonPress-1>", self.begin_drag)
        self.scene_label.bind("<B1-Motion>", self.drag_scene)
        self.scene_label.bind("<ButtonRelease-1>", self.end_drag)
        self.scene_label.bind("<MouseWheel>", self.scroll_zoom)
        self.scene_label.bind("<Button-4>", lambda _event: self.zoom(0.9))
        self.scene_label.bind("<Button-5>", lambda _event: self.zoom(1.1))

        camera_panel = tk.Frame(content)
        camera_panel.pack(side=tk.RIGHT, fill=tk.Y)
        self.camera_labels: dict[str, object] = {}
        self.camera_photos: dict[str, object] = {}
        for name in camera_names:
            panel = tk.Frame(camera_panel, bd=1, relief=tk.GROOVE)
            panel.pack(fill=tk.X, padx=4, pady=4)
            tk.Label(panel, text=name, anchor="w").pack(fill=tk.X)
            label = tk.Label(panel, text="waiting", width=320, height=240)
            label.pack()
            self.camera_labels[name] = label

        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _photo(self, frame: np.ndarray):
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        height, width = frame.shape[:2]
        ppm = f"P6\n{width} {height}\n255\n".encode() + frame.tobytes()
        with tempfile.NamedTemporaryFile(suffix=".ppm") as image_file:
            image_file.write(ppm)
            image_file.flush()
            return self._tk.PhotoImage(file=image_file.name, format="PPM")

    def render(self) -> None:
        if self.closed:
            return
        with self.host.physics_lock:
            self._scene_renderer.update_scene(self.host.data, camera=self._free_camera)
            scene_frame = self._scene_renderer.render()
            camera_frames = [self.host.camera_image(name) for name in self.camera_names]
        scene_photo = self._photo(scene_frame)
        self.scene_photo = scene_photo
        self.scene_label.configure(image=scene_photo, text="")
        for name, label, frame in zip(
            self.camera_names, self.camera_labels.values(), camera_frames
        ):
            photo = self._photo(frame)
            self.camera_photos[name] = photo
            label.configure(image=photo, text="")
        state = self.host.latest_state()
        step = state["step"] if state is not None else 0
        self.paused = self.host.paused
        self.status.configure(
            text=f"{'paused' if self.paused else 'running'}  step={step}"
        )
        self.root.update_idletasks()
        self.root.update()

    def tick(self) -> None:
        if self.closed:
            return
        self.render()
        control_hz = float(
            getattr(getattr(self.host.robot, "cfg", None), "control_hz", 25.0)
        )
        self.root.after(max(1, round(1000 / control_hz)), self.tick)

    def reset(self) -> None:
        self.host.reset()

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self.host.set_paused(self.paused)
        self.pause_button.configure(text="Resume" if self.paused else "Pause")

    def zoom(self, factor: float) -> None:
        self._free_camera.distance = float(
            np.clip(self._free_camera.distance * factor, 0.05, 5.0)
        )

    def begin_drag(self, event) -> None:
        self._last_drag = (event.x, event.y)

    def drag_scene(self, event) -> None:
        if self._last_drag is None:
            return
        last_x, last_y = self._last_drag
        dx, dy = event.x - last_x, event.y - last_y
        self._free_camera.azimuth -= dx * 0.5
        self._free_camera.elevation = float(
            np.clip(self._free_camera.elevation + dy * 0.5, -89.0, 89.0)
        )
        self._last_drag = (event.x, event.y)

    def end_drag(self, _event) -> None:
        self._last_drag = None

    def scroll_zoom(self, event) -> None:
        self.zoom(0.9 if event.delta > 0 else 1.1)

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self._scene_renderer.close()
            self.root.destroy()


def build_policy(name: str, env, checkpoint: Path | None = None):
    if env.robot_spec.supports("base_velocity"):
        if name != "constant":
            raise ValueError("TurtleBot4 currently supports --policy constant only")
        name = "constant_twist"
    if name == "lerobot" and checkpoint is None:
        raise ValueError("--policy lerobot needs --checkpoint")
    policy_kwargs = {"env": env}
    if name == "lerobot":
        policy_kwargs["checkpoint"] = checkpoint
    return create_policy(name, **policy_kwargs)


def build_so101_config(
    args: argparse.Namespace,
    task_config: TaskConfig | None,
    seed: int,
    max_steps: int,
    render: bool,
    domain_randomization: DomainRandomizationConfig,
) -> EnvConfig:
    if task_config is None:
        cam_w, cam_h = (
            (args.camera_size, args.camera_size) if args.camera_size else (640, 480)
        )
        return EnvConfig(
            scene=SceneConfig(camera_width=cam_w, camera_height=cam_h),
            seed=seed,
            max_steps=max_steps,
            render=render,
            domain_randomization=domain_randomization,
        )

    config = task_config.env
    scene = config.scene
    if args.camera_size:
        scene = replace(
            scene,
            camera_width=args.camera_size,
            camera_height=args.camera_size,
        )
    return replace(
        config,
        scene=scene,
        seed=seed,
        max_steps=max_steps,
        render=render,
        domain_randomization=domain_randomization,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sim-config",
        type=Path,
        default=Path("configs/sim_config.yaml"),
        help="shared simulation configuration",
    )
    ap.add_argument(
        "--config",
        type=Path,
        help="YAML task configuration (for example configs/tasks/so101/pick_place.yaml)",
    )
    ap.add_argument(
        "--robot",
        choices=available_robots(),
        help="override the robot selected by --config",
    )
    # "lerobot" belongs here: build_policy() handles it and the module
    # docstring documents it, but dropping it from choices made argparse
    # reject the documented command before it ever got there.
    ap.add_argument(
        "--policy",
        default="scripted",
        choices=[name for name in available_policies() if name != "replay"],
    )
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument(
        "--seed", type=int, help="override the seed selected by --config (default: 0)"
    )
    ap.add_argument(
        "--max-steps", type=int, help="override the episode length selected by --config"
    )
    ap.add_argument("--camera", default="front")
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument(
        "--camera-size",
        type=int,
        help="square render resolution. IMPORTANT for --policy lerobot: "
        "a policy trained on square images (collect_demos.py's "
        "default) sees a stretched, off-distribution image if "
        "you render non-square here — pass the training size "
        "(e.g. 128) to avoid the mismatch.",
    )
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument(
        "--video", action="store_true", help="render frames and write an episode video"
    )
    ap.add_argument("--no-video", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument(
        "--viewer",
        action="store_true",
        help="open the single-window interactive scene and camera viewer",
    )
    ap.add_argument(
        "--serve",
        action="store_true",
        help="serve the same authoritative simulation to the web viewer",
    )
    ap.add_argument("--host", default="127.0.0.1", help="web host bind address")
    ap.add_argument("--port", type=int, default=8000, help="web host port")
    ap.add_argument(
        "--camera-view",
        action="store_true",
        help="compatibility flag; --viewer already shows all cameras",
    )
    args = ap.parse_args()

    if args.serve and not args.viewer:
        ap.error("--serve requires --viewer")

    sim_config = load_sim_config(args.sim_config)
    task_config = load_task_config(args.config) if args.config else None
    configured_robot = task_config.robot if task_config else (args.robot or "so101")
    if args.robot and args.robot != configured_robot:
        ap.error(
            f"--robot {args.robot!r} does not match --config robot {configured_robot!r}"
        )
    args.robot = args.robot or configured_robot
    if task_config and args.robot != "so101":
        ap.error("--config currently supports the SO-101 pick-and-place workflow only")
    seed = (
        args.seed
        if args.seed is not None
        else (
            task_config.env.seed
            if task_config and task_config.env.seed is not None
            else sim_config.seed
        )
    )
    max_steps = (
        args.max_steps
        if args.max_steps is not None
        else (task_config.env.max_steps if task_config else 600)
    )

    if args.viewer:
        if args.camera_view and args.robot == "turtlebot4":
            ap.error("--camera-view currently supports the SO-101 viewer only")
        return run_viewer(
            args, task_config, seed, max_steps, sim_config.domain_randomization
        )

    if args.robot == "turtlebot4":
        env = create_robot(
            args.robot,
            config=TurtleBot4Config(
                max_steps=max_steps,
                render=args.video and not args.no_video,
                domain_randomization=sim_config.domain_randomization,
            ),
        )
        camera_name = "free"
    else:
        robot = create_robot(
            args.robot,
            config=build_so101_config(
                args,
                task_config,
                seed,
                max_steps,
                render=args.video and not args.no_video,
                domain_randomization=sim_config.domain_randomization,
            ),
        )
        env = TaskRuntime(
            robot,
            create_task(
                task_config.task if task_config else "pick_place",
                success_xy_tol=task_config.success_xy_tol if task_config else 0.04,
            ),
            success_hold_steps=task_config.success_hold_steps if task_config else 10,
        )
        camera_name = args.camera
    args.out.mkdir(parents=True, exist_ok=True)
    successes = 0

    # Built once — a lerobot checkpoint is expensive to reload per episode.
    policy = build_policy(args.policy, env, args.checkpoint)

    for ep in range(args.episodes):
        obs = env.reset(seed=seed + ep)
        policy.reset(obs)
        print(f"  randomization={env.randomization_metadata.as_dict()}")
        frames, total_reward, info = [], 0.0, {}

        for _ in range(max_steps):
            if args.video and not args.no_video:
                frames.append(env.render_camera(camera_name))
            obs, reward, terminated, truncated, info = env.step(policy.act(obs))
            total_reward += reward
            if terminated or truncated or policy.done:
                break

        ok = bool(info.get("success"))
        successes += ok
        distance = info.get("dist_cube_target")
        suffix = f" dist_cube_target={distance:.3f}" if distance is not None else ""
        print(
            f"episode {ep}: success={ok} steps={env.step_count} "
            f"return={total_reward:.2f}{suffix}"
        )

        if frames and args.video and not args.no_video:
            path = write_video(
                np.stack(frames),
                args.out / f"{args.policy}_ep{ep:03d}",
                fps=int(env.cfg.control_hz),
            )
            print(f"  video -> {path}")

    env.close()
    print(f"\n{successes}/{args.episodes} successful")
    return 0


def run_viewer(
    args: argparse.Namespace,
    task_config: TaskConfig | None,
    seed: int,
    max_steps: int,
    domain_randomization: DomainRandomizationConfig,
) -> int:
    if args.robot == "turtlebot4":
        env = create_robot(
            args.robot,
            config=TurtleBot4Config(
                max_steps=args.max_steps,
                render=True,
                domain_randomization=domain_randomization,
            ),
        )
        camera_names = ["free"]
    else:
        viewer_config = build_so101_config(
            args,
            task_config,
            seed,
            max_steps,
            render=True,
            domain_randomization=domain_randomization,
        )
        viewer_config = replace(
            viewer_config,
            camera_stride=max(1, round(viewer_config.control_hz / 5)),
        )
        env = create_robot(
            args.robot,
            config=viewer_config,
        )
    policy = build_policy(args.policy, env, args.checkpoint)
    if args.robot != "turtlebot4":
        camera_names = [
            mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id)
            for camera_id in range(env.model.ncam)
        ]
        camera_names = [name for name in camera_names if name]

    host = SimulationHost(
        env,
        robot_name=args.robot,
        policy=policy,
        reset_seed=seed,
    )
    host.start()
    server = None
    server_thread = None
    if args.serve:
        try:
            import uvicorn

            from physai.web.app import create_app
        except ImportError as exc:
            host.stop()
            raise SystemExit(
                "install web dependencies with: uv sync --extra web"
            ) from exc
        server = uvicorn.Server(
            uvicorn.Config(
                create_app(host=host),
                host=args.host,
                port=args.port,
                log_level="info",
            )
        )
        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()

    app = None
    try:
        print("Custom viewer open. Close the window to exit.")
        if args.serve:
            print(f"Web viewer: http://{args.host}:{args.port}/")
        app = SingleWindowViewer(host, camera_names)
        app.tick()
        app.root.mainloop()
    finally:
        if app is not None and not app.closed:
            app.close()
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=2.0)
        host.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
