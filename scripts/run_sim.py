"""Run a session and look at it: a quick check, the viewer or web host, a video, a recording.

For numbers you compare across policies use `eval_policy.py`; both run the session in
a manifest and stop an episode the same way.

python scripts/run_sim.py                      # scripted expert, 1 episode
python scripts/run_sim.py --manifest configs/manifests/so101_single_cube_fixed_place.yaml
python scripts/run_sim.py --episodes 5 --seed 0
python scripts/run_sim.py --video --episodes 5 --seed 0
python scripts/run_sim.py --record --video     # also save every input as data (outputs/)
python scripts/run_sim.py --policy constant    # baseline: do nothing
python scripts/run_sim.py --policy lerobot --checkpoint outputs/act_ckpt
python scripts/run_sim.py --viewer             # native MuJoCo viewer
python scripts/run_sim.py --viewer --serve     # native viewer plus shared web host
python scripts/run_sim.py --serve              # web host only, no desktop window
python scripts/run_sim.py --sim isaac --manifest configs/manifests/so101_single_cube_fixed_place.yaml --policy visual_servo
                                                # the same manifest on Isaac Sim; --video, --record
                                                # and --serve (web viewer) work as on MuJoCo
                                                # (no --viewer/--dataset-dir with --serve; needs
                                                # isaacsim installed, see README.md; not exercised
                                                # by this repo's own CI)

A run is described by a session manifest (`--manifest`); a bare `--robot` is converted
into one, so every run takes the same path from there on.
"""

from __future__ import annotations

import argparse
import signal
import threading
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from _common_args import (
    add_camera_resolution,
    add_episodes,
    add_max_steps,
    add_policy,
    add_policy_args,
    add_robot,
    add_run_outputs,
    add_seed,
    add_simulator,
    policy_kwargs,
)

# Registers so101's "scripted"/"visual_servo" policies and the checkpoint-
# backed "lerobot" policy with their registries; --policy may select any of
# them, so all load eagerly (none import torch/lerobot at module scope).
import research.classical_control.so101_visual_servo
import research.imitation_learning.vla_adapter
import research.scripted_experts.so101_pick_place_expert  # noqa: F401
from _outputs import RunOutputs, wants_cameras
from physai.config import SessionManifest, load_manifest, load_sim_config
from physai.config.compat import manifest_for_robot, with_overrides
from physai.policy import available_policies
from physai.robots import available_robots
from physai.runtime import RenderGlitch, Session, create_session, run_episode
from physai.web.host import Host

DEFAULT_SIM_CONFIG = Path("configs/sim_config.yaml")
# What a headless episode runs when neither --policy nor the manifest names one.
DEFAULT_HEADLESS_POLICY = "scripted"


def parse_args(
    argv: list[str] | None = None,
) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        type=Path,
        help="session manifest YAML (for example configs/manifests/so101_single_cube_fixed_place.yaml)",
    )
    ap.add_argument(
        "--sim-config",
        type=Path,
        help="shared simulation settings for --robot "
        f"(default: {DEFAULT_SIM_CONFIG}); a manifest has its own",
    )
    add_robot(
        ap,
        choices=available_robots(),
        help="the robot to run when no manifest selects one",
    )
    add_simulator(ap)
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
    add_policy_args(ap)
    add_camera_resolution(ap)
    add_run_outputs(ap)
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
    return ap, ap.parse_args(argv)


def build_manifest(
    ap: argparse.ArgumentParser, args: argparse.Namespace
) -> SessionManifest:
    """The session the flags describe, with command-line overrides applied."""
    if args.manifest and (args.sim_config or args.robot):
        ap.error("--manifest cannot be combined with --sim-config or --robot")

    if args.manifest:
        manifest = load_manifest(args.manifest)
    else:
        simulation = load_sim_config(args.sim_config or DEFAULT_SIM_CONFIG)
        manifest = manifest_for_robot(args.robot or "so101", simulation=simulation)

    if manifest.world is not None and args.policy:
        ap.error("--policy cannot be used with a shared world")
    return with_overrides(
        manifest,
        seed=args.seed,
        max_steps=args.max_steps,
        camera_resolution=args.camera_resolution,
        policy=args.policy,
        simulator=args.simulator,
    )


def main(argv: list[str] | None = None) -> int:
    ap, args = parse_args(argv)
    if args.record and args.serve:
        ap.error("--record writes per-episode files; with --serve use --dataset-dir")

    manifest = build_manifest(ap, args)
    if manifest.world is not None and not (args.viewer or args.serve):
        ap.error("a shared-world session requires --viewer or --serve")
    if args.dataset_dir and manifest.world is not None:
        ap.error("--dataset-dir is not available with a shared world")
    if manifest.simulator != "mujoco":
        if args.viewer:
            ap.error(
                f"--viewer is MuJoCo-only; simulator {manifest.simulator!r} "
                "supports --serve (web viewer) or headless episodes"
            )
        if args.serve and args.dataset_dir:
            ap.error("--dataset-dir is MuJoCo-only")

    if args.viewer or args.serve:
        return run_viewer(args, manifest)
    return run_episodes(args, manifest)


def run_episodes(args: argparse.Namespace, manifest: SessionManifest) -> int:
    if manifest.policy_for(manifest.robots[0]) == "idle":
        if manifest.simulator != "mujoco":
            raise SystemExit(
                f"--policy is required for simulator {manifest.simulator!r}: the "
                f"default headless policy ({DEFAULT_HEADLESS_POLICY!r}) needs "
                "MuJoCo-only kinematics (ArmKinematics)"
            )
        manifest = with_overrides(manifest, policy=DEFAULT_HEADLESS_POLICY)
    policy_name = manifest.policy_for(manifest.robots[0])
    # Built once — a lerobot checkpoint is expensive to reload per episode.
    session = create_session(
        manifest,
        render=wants_cameras(args),
        policy_kwargs=policy_kwargs(args, policy_name),
    )
    runtime = session.runtime
    env = runtime.robot
    seed = manifest.simulation.seed
    outputs = RunOutputs(
        runtime,
        args,
        simulator=manifest.simulator,
        robot=manifest.robots[0].robot,
        policy=policy_name,
        task=policy_name,
        fps=env.cfg.control_hz,
    )
    successes = 0

    for ep in range(args.episodes):
        try:
            outcome = run_episode(runtime, seed + ep, outputs.begin())
        except RenderGlitch as exc:
            session.close()
            raise SystemExit(f"episode {ep}: {exc}; restart the run") from exc
        randomization = getattr(env, "randomization_metadata", None)
        if randomization is not None:
            print(f"  randomization={randomization.as_dict()}")

        successes += outcome.success
        distance = outcome.info.get("dist_cube_target")
        suffix = f" dist_cube_target={distance:.3f}" if distance is not None else ""
        refused = f" UNSAFE: {outcome.violation}" if outcome.violation else ""
        print(
            f"episode {ep}: success={outcome.success} steps={outcome.steps} "
            f"return={outcome.reward:.2f}{suffix}{refused}"
        )

        outputs.save(outcome.success, seed + ep)

    outputs.close()
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
        policy_kwargs=(
            {}
            if manifest.world is not None
            else policy_kwargs(args, manifest.policy_for(manifest.robots[0]))
        ),
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
        dataset_dir=args.dataset_dir,
    )
    return host, session


def serve_on_own_loop(server) -> None:
    """Run uvicorn without `asyncio.run`, which Isaac Sim's Kit app replaces
    with a version that rejects uvicorn's `loop_factory` argument."""
    import asyncio

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(server.serve())
    finally:
        loop.close()


def run_viewer(args: argparse.Namespace, manifest: SessionManifest) -> int:
    if args.viewer:
        import mujoco
        import mujoco.viewer
    host, _session = build_host(args, manifest)
    if manifest.simulator == "mujoco":
        host.start()
    server = None
    server_thread = None
    if args.serve:
        try:
            import uvicorn

            from physai.web.app import create_app
        except ImportError as exc:
            if manifest.simulator == "mujoco":
                host.stop()
            raise SystemExit(
                "fastapi/uvicorn are missing; reinstall with: uv sync"
            ) from exc
        server = uvicorn.Server(
            uvicorn.Config(
                create_app(host=host),
                host=args.host,
                port=args.port,
                log_level="info",
            )
        )
        server_thread = threading.Thread(
            target=server.run if manifest.simulator == "mujoco" else serve_on_own_loop,
            args=() if manifest.simulator == "mujoco" else (server,),
            daemon=True,
        )
        server_thread.start()

    try:
        if manifest.simulator != "mujoco":
            # Isaac Sim must be driven from the thread that created it, so
            # the host loop takes the main thread; signals just stop it.
            print(f"Isaac host running. Web viewer: http://{args.host}:{args.port}/")
            print("Press Ctrl+C to stop.")
            signal.signal(signal.SIGINT, lambda *_: host.stop())
            signal.signal(signal.SIGTERM, lambda *_: host.stop())
            host.run()
        elif not args.viewer:
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
