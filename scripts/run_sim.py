"""Run one episode and write a video. The 30-second sanity check.

    python scripts/run_sim.py                      # scripted expert, 1 episode
    python scripts/run_sim.py --episodes 5 --seed 0
    python scripts/run_sim.py --policy constant    # baseline: do nothing
    python scripts/run_sim.py --policy lerobot --checkpoint outputs/act_ckpt
    python scripts/run_sim.py --viewer             # interactive MuJoCo viewer
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from physai.contracts import Action, Twist
from physai.policy import ConstantPolicy, ScriptedPickPlace
from physai.robots import available_robots, create_robot
from physai.robots.turtlebot import TurtleBot4Config
from physai.sim import EnvConfig, SceneConfig


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
    print("  (no H.264 encoder found — wrote a GIF; `pip install imageio-ffmpeg` for mp4)")
    return gif


def build_policy(name: str, env, checkpoint: Path | None = None):
    if env.robot_spec.supports("base_velocity"):
        if name != "constant":
            raise ValueError("TurtleBot4 currently supports --policy constant only")
        return ConstantTwistPolicy()
    if name == "scripted":
        return ScriptedPickPlace(env.kin, env)
    if name == "constant":
        return ConstantPolicy()
    if name == "lerobot":
        if checkpoint is None:
            raise ValueError("--policy lerobot needs --checkpoint")
        return LeRobotPolicy.from_checkpoint(env, checkpoint)
    raise ValueError(f"unknown policy {name!r}")


class ConstantTwistPolicy:
    """Hold a mobile base still for the generic simulation smoke test."""

    def reset(self, observation, goal=None, instruction=None) -> None:
        pass

    def act(self, observation) -> Action:
        return Action(ee_twist=Twist())

    @property
    def done(self) -> bool:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="so101", choices=available_robots())
    ap.add_argument("--policy", default="scripted", choices=["scripted", "constant"])
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=600)
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

    if args.viewer:
        return run_viewer(args)

    if args.robot == "turtlebot4":
        env = create_robot(args.robot, config=TurtleBot4Config(
            max_steps=args.max_steps, render=not args.no_video,
        ))
        camera_name = "free"
    else:
        env = create_robot(args.robot, config=EnvConfig(
            scene=SceneConfig(camera_width=640, camera_height=480),
            seed=args.seed, max_steps=args.max_steps, render=True,
        ))
        camera_name = args.camera
    args.out.mkdir(parents=True, exist_ok=True)
    successes = 0

    # Built once — a lerobot checkpoint is expensive to reload per episode.
    policy = build_policy(args.policy, env, args.checkpoint)

    for ep in range(args.episodes):
        obs = env.reset(seed=args.seed + ep)
        policy.reset(obs)
        frames, total_reward, info = [], 0.0, {}

        for _ in range(args.max_steps):
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


def run_viewer(args) -> int:
    import mujoco.viewer

    if args.robot == "turtlebot4":
        env = create_robot(args.robot, config=TurtleBot4Config(
            max_steps=args.max_steps, render=False,
        ))
    else:
        env = create_robot(args.robot, config=EnvConfig(
            seed=args.seed, render=False, max_steps=args.max_steps,
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
