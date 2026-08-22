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

from physai.policy import ConstantPolicy, LeRobotPolicy, ScriptedPickPlace
from physai.sim import EnvConfig, SO101PickPlaceEnv, SceneConfig


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
    if name == "scripted":
        return ScriptedPickPlace(env.kin, env)
    if name == "constant":
        return ConstantPolicy()
    if name == "lerobot":
        if checkpoint is None:
            raise ValueError("--policy lerobot needs --checkpoint")
        return LeRobotPolicy.from_checkpoint(env, checkpoint)
    raise ValueError(f"unknown policy {name!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="scripted",
                    choices=["scripted", "constant", "lerobot"])
    ap.add_argument("--checkpoint", type=Path, help="required for --policy lerobot")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--camera", default="front")
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

    cam_w, cam_h = (args.camera_size, args.camera_size) if args.camera_size else (640, 480)
    env = SO101PickPlaceEnv(EnvConfig(
        scene=SceneConfig(camera_width=cam_w, camera_height=cam_h),
        seed=args.seed, max_steps=args.max_steps, render=True,
    ))
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
                frames.append(env.render_camera(args.camera))
            obs, reward, terminated, truncated, info = env.step(policy.act(obs))
            total_reward += reward
            if terminated or truncated or policy.done:
                break

        ok = bool(info.get("success"))
        successes += ok
        print(f"episode {ep}: success={ok} steps={env.step_count} "
              f"return={total_reward:.2f} dist_cube_target={info['dist_cube_target']:.3f}")

        if frames and not args.no_video:
            path = write_video(np.stack(frames), args.out / f"{args.policy}_ep{ep:03d}",
                               fps=int(env.cfg.control_hz))
            print(f"  video -> {path}")

    env.close()
    print(f"\n{successes}/{args.episodes} successful")
    return 0


def run_viewer(args) -> int:
    import mujoco.viewer

    env = SO101PickPlaceEnv(EnvConfig(seed=args.seed, render=args.policy == "lerobot",
                                      max_steps=args.max_steps))
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
