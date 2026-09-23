"""Run one episode, optionally writing a video. The 30-second sanity check.

python scripts/run_sim.py                      # scripted expert, 1 episode
python scripts/run_sim.py --config configs/tasks/so101/pick_place.yaml
python scripts/run_sim.py --episodes 5 --seed 0
python scripts/run_sim.py --video --episodes 5 --seed 0
python scripts/run_sim.py --policy constant    # baseline: do nothing
python scripts/run_sim.py --policy lerobot --checkpoint outputs/act_ckpt
python scripts/run_sim.py --viewer             # native MuJoCo viewer
python scripts/run_sim.py --viewer --serve     # native viewer plus shared web host
python scripts/run_sim.py --serve              # web host only, no desktop window
"""

from __future__ import annotations

import argparse
import signal
import threading
import time
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401
import mujoco
import mujoco.viewer
import numpy as np
from _common_args import (
    add_checkpoint,
    add_episodes,
    add_max_steps,
    add_out,
    add_policy,
    add_robot,
    add_seed,
)

from physai.config import (
    DomainRandomizationConfig,
    TaskConfig,
    load_sim_config,
    load_task_config,
    load_world_config,
)
from physai.policy import available_policies, create_policy
from physai.robots import available_robots, create_robot, shared_attach
from physai.robots.so101 import EnvConfig
from physai.robots.turtlebot import TurtleBot4Config
from physai.sim import PickPlaceMinimalSceneConfig, SharedWorld
from physai.tasks import TaskRuntime, create_task
from physai.web.host import Host

# Registers so101's "scripted"/"visual_servo" policies and the checkpoint-
# backed "lerobot" policy with their registries; --policy may select any of
# them, so all load eagerly (none import torch/lerobot at module scope).
import research.classical_control.so101_visual_servo  # noqa: E402,F401
import research.imitation_learning.vla_adapter  # noqa: E402,F401
import research.scripted_experts.so101_pick_place_expert  # noqa: E402,F401


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
            scene=PickPlaceMinimalSceneConfig(camera_width=cam_w, camera_height=cam_h),
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
        "--world",
        type=Path,
        help="shared-world YAML manifest; use with --viewer and/or --serve",
    )
    add_robot(
        ap, choices=available_robots(), help="override the robot selected by --config"
    )
    # "lerobot" belongs here: build_policy() handles it and the module
    # docstring documents it, but dropping it from choices made argparse
    # reject the documented command before it ever got there.
    add_policy(
        ap,
        choices=[name for name in available_policies() if name != "replay"],
        help="policy to run; viewer/serve stays idle unless this is specified",
    )
    add_episodes(ap, default=1)
    add_seed(
        ap, default=None, help="override the seed selected by --config (default: 0)"
    )
    add_max_steps(ap, help="override the episode length selected by --config")
    ap.add_argument("--camera", default="front")
    add_checkpoint(ap)
    ap.add_argument(
        "--camera-size",
        type=int,
        help="square render resolution. IMPORTANT for --policy lerobot: "
        "a policy trained on square images (collect_demos.py's "
        "default) sees a stretched, off-distribution image if "
        "you render non-square here — pass the training size "
        "(e.g. 128) to avoid the mismatch.",
    )
    add_out(ap, default=Path("outputs"))
    ap.add_argument(
        "--video", action="store_true", help="render frames and write an episode video"
    )
    ap.add_argument(
        "--viewer",
        action="store_true",
        help="open MuJoCo's native interactive scene viewer",
    )
    ap.add_argument(
        "--serve",
        action="store_true",
        help="serve the same authoritative simulation to the web viewer; "
        "without --viewer, this runs with no desktop window",
    )
    ap.add_argument("--host", default="127.0.0.1", help="web host bind address")
    ap.add_argument("--port", type=int, default=8000, help="web host port")
    args = ap.parse_args()

    if args.world and not (args.viewer or args.serve):
        ap.error("--world requires --viewer or --serve")
    if args.world and (args.config or args.robot):
        ap.error("--world cannot be combined with --config or --robot")

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

    if args.viewer or args.serve:
        return run_viewer(
            args, task_config, seed, max_steps, sim_config.domain_randomization
        )

    if args.robot == "turtlebot4":
        env = create_robot(
            args.robot,
            config=TurtleBot4Config(
                max_steps=max_steps,
                render=args.video,
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
                render=args.video,
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
    policy = build_policy(args.policy or "scripted", env, args.checkpoint)

    for ep in range(args.episodes):
        obs = env.reset(seed=seed + ep)
        policy.reset(obs)
        print(f"  randomization={env.randomization_metadata.as_dict()}")
        frames, total_reward, info = [], 0.0, {}

        for _ in range(max_steps):
            if args.video:
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

        if frames and args.video:
            path = write_video(
                np.stack(frames),
                args.out / f"{args.policy}_ep{ep:03d}",
                fps=int(env.cfg.control_hz),
            )
            print(f"  video -> {path}")

    env.close()
    print(f"\n{successes}/{args.episodes} successful")
    return 0


def wait_for_shutdown() -> None:
    """Block the main thread until Ctrl+C or SIGTERM.

    Containers and process managers stop a service with SIGTERM, so it is
    handled like Ctrl+C: the caller's ``finally`` block then stops the web
    server and the physics host in order instead of being killed mid-render.
    The wait is polled with a timeout because an untimed ``Event.wait`` is not
    reliably interrupted by signals on Windows.
    """
    stop = threading.Event()

    def request_stop(signum, frame) -> None:
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    while not stop.wait(0.5):
        pass


def run_viewer(
    args: argparse.Namespace,
    task_config: TaskConfig | None,
    seed: int,
    max_steps: int,
    domain_randomization: DomainRandomizationConfig,
) -> int:
    if args.world:
        world_config = load_world_config(args.world)
        host = Host.for_world(
            SharedWorld(
                world_config.instances,
                timestep=world_config.timestep,
                control_hz=world_config.control_hz,
                add_floor=world_config.add_floor,
                shared_attach=shared_attach,
            ),
            world_config.instances,
        )
    elif args.robot == "turtlebot4":
        env = create_robot(
            args.robot,
            config=TurtleBot4Config(
                max_steps=args.max_steps,
                render=True,
                domain_randomization=domain_randomization,
            ),
        )
    else:
        viewer_config = build_so101_config(
            args,
            task_config,
            seed,
            max_steps,
            render=True,
            domain_randomization=domain_randomization,
        )
        # camera_stride=0: the env never renders cameras inline on the
        # physics thread in interactive mode, for any policy. Rendering is
        # comparatively expensive, so doing it inline stalled physics
        # stepping every stride'th tick; the async camera thread below
        # (async_cameras=True) is the one source of camera frames instead —
        # Host._sync_observation_images() feeds a policy that needs vision
        # from that same cache. This is now the standard for so101 in
        # --viewer/--serve, not just the idle/scripted case.
        viewer_config = replace(viewer_config, camera_stride=0)
        env = create_robot(
            args.robot,
            config=viewer_config,
        )
    if not args.world:
        policy = (
            build_policy(args.policy, env, args.checkpoint)
            if args.policy is not None
            else None
        )
        host = Host.for_robot(
            env,
            robot_name=args.robot,
            policy=policy,
            reset_seed=seed,
            async_cameras=args.robot == "so101",
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

    try:
        if not args.viewer:
            # No desktop window: --serve alone runs headless.
            print(f"Headless host running. Web viewer: http://{args.host}:{args.port}/")
            print("Press Ctrl+C to stop.")
            wait_for_shutdown()
        else:
            print("MuJoCo viewer open. Close the window to exit.")
            if args.serve:
                print(f"Web viewer: http://{args.host}:{args.port}/")
            # Host.start() steps physics on its own thread. launch_passive's
            # own render thread reads its mjData continuously, not only at
            # sync() — sharing host.data directly races that render thread
            # against Host's physics/camera threads (both guarded by
            # host.physics_lock, which the native viewer knows nothing
            # about). Mirror Host._camera_loop's pattern instead: render a
            # private copy, refreshed each tick under the same lock (ADR 4).
            viewer_data = mujoco.MjData(host.model)
            with mujoco.viewer.launch_passive(host.model, viewer_data) as viewer:
                period = 1.0 / host.control_hz
                while viewer.is_running():
                    with host.physics_lock:
                        mujoco.mj_copyData(viewer_data, host.model, host.data)
                    viewer.sync()
                    time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=2.0)
        host.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
