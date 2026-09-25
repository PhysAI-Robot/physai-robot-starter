"""Run one episode, optionally writing a video. The 30-second sanity check.

python scripts/run_sim.py                      # scripted expert, 1 episode
python scripts/run_sim.py --manifest configs/manifests/so101_pick_place.yaml
python scripts/run_sim.py --episodes 5 --seed 0
python scripts/run_sim.py --video --episodes 5 --seed 0
python scripts/run_sim.py --policy constant    # baseline: do nothing
python scripts/run_sim.py --policy lerobot --checkpoint outputs/act_ckpt
python scripts/run_sim.py --viewer             # native MuJoCo viewer
python scripts/run_sim.py --viewer --serve     # native viewer plus shared web host
python scripts/run_sim.py --serve              # web host only, no desktop window

A run is described by a session manifest (`--manifest`). The older `--config`
(task file), `--world` (world file), and bare `--robot` inputs are converted
into one, so every run takes the same path from there on.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
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

# Registers so101's "scripted"/"visual_servo" policies and the checkpoint-
# backed "lerobot" policy with their registries; --policy may select any of
# them, so all load eagerly (none import torch/lerobot at module scope).
import research.classical_control.so101_visual_servo
import research.imitation_learning.vla_adapter
import research.scripted_experts.so101_pick_place_expert  # noqa: F401
from physai.config import SessionManifest, load_manifest, load_sim_config
from physai.config.compat import (
    manifest_for_robot,
    manifest_from_task_file,
    manifest_from_world_file,
    with_overrides,
)
from physai.policy import available_policies
from physai.robots import available_robots
from physai.runtime import Session, create_session
from physai.web.host import Host

DEFAULT_SIM_CONFIG = Path("configs/sim_config.yaml")
# What a headless episode runs when neither --policy nor the manifest names one.
DEFAULT_HEADLESS_POLICY = "scripted"


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


def parse_args(
    argv: list[str] | None = None,
) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        type=Path,
        help="session manifest YAML (for example configs/manifests/so101_pick_place.yaml)",
    )
    ap.add_argument(
        "--sim-config",
        type=Path,
        help="shared simulation settings for --config/--world/--robot "
        f"(default: {DEFAULT_SIM_CONFIG}); a manifest has its own",
    )
    ap.add_argument(
        "--config",
        type=Path,
        help="deprecated: task YAML (for example configs/tasks/so101/pick_place.yaml)",
    )
    ap.add_argument(
        "--world",
        type=Path,
        help="deprecated: shared-world YAML; use with --viewer and/or --serve",
    )
    add_robot(
        ap, choices=available_robots(), help="the robot to run when no file selects one"
    )
    # "lerobot" belongs here: main() handles it and the module docstring
    # documents it, but dropping it from choices made argparse reject the
    # documented command before it ever got there.
    add_policy(
        ap,
        choices=[name for name in available_policies() if name != "replay"],
        help="policy to run; viewer/serve stays idle unless this is specified",
    )
    add_episodes(ap, default=1)
    add_seed(ap, default=None, help="override the seed (default: the manifest's, or 0)")
    add_max_steps(ap, help="override the episode length")
    ap.add_argument(
        "--camera",
        help="camera to record with --video (default: the robot's first camera)",
    )
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
    ap.add_argument(
        "--record-dir",
        type=Path,
        help="enable browser episode recording into this dataset directory "
        "(requires --serve; an existing dataset there is continued)",
    )
    ap.add_argument("--host", default="127.0.0.1", help="web host bind address")
    ap.add_argument("--port", type=int, default=8000, help="web host port")
    return ap, ap.parse_args(argv)


def build_manifest(
    ap: argparse.ArgumentParser, args: argparse.Namespace
) -> SessionManifest:
    """The session the flags describe, with command-line overrides applied."""
    sources = [flag for flag in ("manifest", "config", "world") if getattr(args, flag)]
    if len(sources) > 1:
        ap.error(f"--{' and --'.join(sources)} cannot be combined")
    if args.manifest and (args.sim_config or args.robot):
        ap.error("--manifest cannot be combined with --sim-config or --robot")
    if args.world and args.robot:
        ap.error("--world cannot be combined with --robot")

    if args.manifest:
        manifest = load_manifest(args.manifest)
    else:
        simulation = load_sim_config(args.sim_config or DEFAULT_SIM_CONFIG)
        if args.world:
            note_deprecated("--world", "configs/manifests/heterogeneous_world.yaml")
            manifest = manifest_from_world_file(args.world, simulation=simulation)
        elif args.config:
            note_deprecated("--config", "configs/manifests/so101_pick_place.yaml")
            manifest = manifest_from_task_file(args.config, simulation=simulation)
            configured = manifest.robots[0].robot
            if args.robot and args.robot != configured:
                ap.error(
                    f"--robot {args.robot!r} does not match --config robot "
                    f"{configured!r}"
                )
        else:
            manifest = manifest_for_robot(args.robot or "so101", simulation=simulation)

    if manifest.world is not None and args.policy:
        ap.error("--policy cannot be used with a shared world")
    return with_overrides(
        manifest,
        seed=args.seed,
        max_steps=args.max_steps,
        camera_size=args.camera_size,
        policy=args.policy,
    )


def note_deprecated(flag: str, replacement: str) -> None:
    print(
        f"warning: {flag} is deprecated; describe the run with --manifest "
        f"(see {replacement})",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    ap, args = parse_args(argv)
    if args.record_dir and not args.serve:
        ap.error("--record-dir requires --serve")

    manifest = build_manifest(ap, args)
    if manifest.world is not None and not (args.viewer or args.serve):
        ap.error("a shared-world session requires --viewer or --serve")
    if args.record_dir and manifest.world is not None:
        ap.error("--record-dir is not available with a shared world")

    if args.viewer or args.serve:
        return run_viewer(args, manifest)
    return run_episodes(args, manifest)


def policy_inputs(args: argparse.Namespace, manifest: SessionManifest) -> dict:
    """Run-time inputs a manifest cannot hold, for the policy it names."""
    if manifest.world is not None:
        return {}
    if manifest.policy_for(manifest.robots[0]) != "lerobot":
        return {}
    if args.checkpoint is None:
        raise ValueError("--policy lerobot needs --checkpoint")
    return {"checkpoint": args.checkpoint}


def run_episodes(args: argparse.Namespace, manifest: SessionManifest) -> int:
    if manifest.policy_for(manifest.robots[0]) == "idle":
        manifest = with_overrides(manifest, policy=DEFAULT_HEADLESS_POLICY)
    # Built once — a lerobot checkpoint is expensive to reload per episode.
    session = create_session(
        manifest, render=args.video, policy_kwargs=policy_inputs(args, manifest)
    )
    runtime = session.runtime
    env, policy = runtime.robot, runtime.policy
    seed = manifest.simulation.seed
    max_steps = env.cfg.max_steps
    camera_name = args.camera or next(iter(env.robot_spec.camera_frames))
    policy_name = manifest.policy_for(manifest.robots[0])
    args.out.mkdir(parents=True, exist_ok=True)
    successes = 0

    for ep in range(args.episodes):
        obs = runtime.reset(seed=seed + ep)
        print(f"  randomization={env.randomization_metadata.as_dict()}")
        frames, total_reward, info = [], 0.0, {}

        for _ in range(max_steps):
            if args.video:
                frames.append(env.render_camera(camera_name))
            obs, reward, terminated, truncated, info = runtime.step(policy.act(obs))
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
                args.out / f"{policy_name}_ep{ep:03d}",
                fps=int(env.cfg.control_hz),
            )
            print(f"  video -> {path}")

    session.close()
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


def build_host(
    args: argparse.Namespace, manifest: SessionManifest
) -> tuple[Host, Session]:
    session = create_session(
        manifest,
        render=True,
        host_driven=True,
        policy_kwargs=policy_inputs(args, manifest),
    )
    if session.world is not None:
        return Host.for_world(session.world, session.instances), session
    robot = session.runtime.robot
    host = Host.for_robot(
        robot,
        robot_name=session.robot_name,
        policy=session.runtime.policy,
        reset_seed=manifest.simulation.seed,
        async_cameras=session.host_renders_cameras,
        record_dir=args.record_dir,
    )
    return host, session


def run_viewer(args: argparse.Namespace, manifest: SessionManifest) -> int:
    host, _session = build_host(args, manifest)
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
