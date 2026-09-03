"""Run one episode and write a video. The 30-second sanity check.

    python scripts/run_sim.py                      # scripted expert, 1 episode
    python scripts/run_sim.py --config configs/task_pick_place.yaml
    python scripts/run_sim.py --episodes 5 --seed 0
    python scripts/run_sim.py --policy constant    # baseline: do nothing
    python scripts/run_sim.py --policy lerobot --checkpoint outputs/act_ckpt
    python scripts/run_sim.py --viewer             # interactive MuJoCo viewer
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from physai.config import TaskConfig, load_sim_config, load_task_config
from physai.policy import available_policies, create_policy
from physai.robots import available_robots, create_robot
from physai.robots.so101 import EnvConfig
from physai.robots.turtlebot import TurtleBot4Config
from physai.sim import SceneConfig
from physai.tasks import TaskRuntime, create_task


def write_video(frames: np.ndarray, stem: Path, fps: int) -> Path:
    """Write mp4 if an H.264 encoder is available, otherwise fall back to GIF.

    imageio's default pyav path raises an unhelpful `expected bytes, NoneType`
    when no codec is registered, so the codec is named explicitly and the
    fallback is silent-but-reported rather than a stack trace.
    """
    import imageio.v3 as iio

    mp4 = stem.with_suffix(".mp4")
    for plugin, kwargs in (("FFMPEG", {"codec": "libx264"}),
                           ("pyav", {"codec": "libx264"})):
        try:
            iio.imwrite(mp4, frames, fps=fps, plugin=plugin, **kwargs)
            return mp4
        except Exception:
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
    return create_policy(name, env=env, checkpoint=checkpoint)


def build_so101_config(
    args: argparse.Namespace,
    task_config: TaskConfig | None,
    seed: int,
    max_steps: int,
    render: bool,
) -> EnvConfig:
    if task_config is None:
        cam_w, cam_h = ((args.camera_size, args.camera_size) if args.camera_size
                        else (640, 480))
        return EnvConfig(
            scene=SceneConfig(camera_width=cam_w, camera_height=cam_h),
            seed=seed,
            max_steps=max_steps,
            render=render,
        )

    config = task_config.env
    scene = config.scene
    if args.camera_size:
        scene = replace(
            scene,
            camera_width=args.camera_size,
            camera_height=args.camera_size,
        )
    return replace(config, scene=scene, seed=seed, max_steps=max_steps, render=render)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-config", type=Path,
                    default=Path("configs/sim_config.yaml"),
                    help="shared simulation configuration")
    ap.add_argument("--config", type=Path,
                    help="YAML task configuration (for example configs/task_pick_place.yaml)")
    ap.add_argument("--robot", choices=available_robots(),
                    help="override the robot selected by --config")
    # "lerobot" belongs here: build_policy() handles it and the module
    # docstring documents it, but dropping it from choices made argparse
    # reject the documented command before it ever got there.
    ap.add_argument(
        "--policy",
        default="scripted",
        choices=[name for name in available_policies() if name != "replay"],
    )
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int,
                    help="override the seed selected by --config (default: 0)")
    ap.add_argument("--max-steps", type=int,
                    help="override the episode length selected by --config")
    ap.add_argument("--camera", default="front")
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--camera-size", type=int,
                    help="square render resolution. IMPORTANT for --policy lerobot: "
                         "a policy trained on square images (collect_demos.py's "
                         "default) sees a stretched, off-distribution image if "
                         "you render non-square here — pass the training size "
                         "(e.g. 128) to avoid the mismatch.")
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--viewer", action="store_true",
                    help="open the interactive viewer instead of writing a video")
    args = ap.parse_args()

    sim_config = load_sim_config(args.sim_config)
    if sim_config.domain_randomization.enabled:
        ap.error(
            "domain randomization is not implemented yet; "
            "set domain_randomization.enabled to false"
        )
    task_config = load_task_config(args.config) if args.config else None
    configured_robot = task_config.robot if task_config else (args.robot or "so101")
    if args.robot and args.robot != configured_robot:
        ap.error(
            f"--robot {args.robot!r} does not match --config robot {configured_robot!r}"
        )
    args.robot = args.robot or configured_robot
    if task_config and args.robot != "so101":
        ap.error("--config currently supports the SO-101 pick-and-place workflow only")
    seed = args.seed if args.seed is not None else (
        task_config.env.seed if task_config and task_config.env.seed is not None
        else sim_config.seed
    )
    max_steps = args.max_steps if args.max_steps is not None else (
        task_config.env.max_steps if task_config else 600
    )

    if args.viewer:
        return run_viewer(args, task_config, seed, max_steps)

    if args.robot == "turtlebot4":
        env = create_robot(args.robot, config=TurtleBot4Config(
            max_steps=max_steps, render=not args.no_video,
        ))
        camera_name = "free"
    else:
        robot = create_robot(args.robot, config=build_so101_config(
            args, task_config, seed, max_steps, render=not args.no_video,
        ))
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
        frames, total_reward, info = [], 0.0, {}

        for _ in range(max_steps):
            if not args.no_video:
                frames.append(env.render_camera(camera_name))
            obs, reward, terminated, truncated, info = env.step(policy.act(obs))
            total_reward += reward
            if terminated or truncated or policy.done:
                break

        ok = bool(info.get("success"))
        successes += ok
        distance = info.get("dist_cube_target")
        suffix = f" dist_cube_target={distance:.3f}" if distance is not None else ""
        print(f"episode {ep}: success={ok} steps={env.step_count} "
              f"return={total_reward:.2f}{suffix}")

        if frames and not args.no_video:
            path = write_video(np.stack(frames), args.out / f"{args.policy}_ep{ep:03d}",
                               fps=int(env.cfg.control_hz))
            print(f"  video -> {path}")

    env.close()
    print(f"\n{successes}/{args.episodes} successful")
    return 0


def run_viewer(
    args: argparse.Namespace,
    task_config: TaskConfig | None,
    seed: int,
    max_steps: int,
) -> int:
    import mujoco.viewer

    if args.robot == "turtlebot4":
        env = create_robot(args.robot, config=TurtleBot4Config(
            max_steps=args.max_steps, render=False,
        ))
    else:
        env = create_robot(args.robot, config=build_so101_config(
            args, task_config, seed, max_steps, render=False,
        ))
    obs = env.reset()
    policy = build_policy(args.policy, env, args.checkpoint)
    policy.reset(obs)

    print("Viewer open. Close the window to exit.")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            obs, _, terminated, truncated, _ = env.step(policy.act(obs))
            viewer.sync()
            if terminated or truncated:
                obs = env.reset()
                policy.reset(obs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
