"""Generate demonstrations with the scripted expert.

This is the dataset you fine-tune a VLA on. Failed episodes are discarded by
default — behaviour cloning on failures teaches failure.

    python scripts/collect_demos.py --episodes 50 --out data/pickplace_v1
    python scripts/collect_demos.py --episodes 50 --keep-failures   # for analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from physai.data import EpisodeRecorder
from physai.policy import ScriptedPickPlace
from physai.robots import available_robots, create_robot
from physai.sim import EnvConfig, SceneConfig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="so101", choices=["so101"],
                    help="collect_demos currently supports the SO-101 manipulation workflow")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("data/pickplace_v1"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--width", type=int, default=224)
    ap.add_argument("--height", type=int, default=224)
    ap.add_argument("--keep-failures", action="store_true")
    ap.add_argument("--no-images", action="store_true",
                    help="record state/action only (much smaller files)")
    ap.add_argument("--task", default="put the red cube on the green pad")
    args = ap.parse_args()

    env = create_robot(args.robot, config=EnvConfig(
        scene=SceneConfig(camera_width=args.width, camera_height=args.height),
        seed=args.seed, max_steps=args.max_steps, render=not args.no_images,
    ))
    rec = EpisodeRecorder(args.out, task=args.task, fps=env.cfg.control_hz,
                          store_images=not args.no_images)

    attempted = kept = 0
    while kept < args.episodes:
        seed = args.seed + attempted
        attempted += 1
        obs = env.reset(seed=seed)
        policy = ScriptedPickPlace(env.kin, env)
        policy.reset(obs)
        rec.start_episode()
        info: dict = {}

        for _ in range(args.max_steps):
            action = policy.act(obs)
            grip_rad = env.gripper_to_joint(action.gripper.clipped())
            prev_obs = obs
            obs, reward, terminated, truncated, info = env.step(action)
            rec.record(prev_obs, action, reward=reward,
                       done=terminated or truncated, phase=policy.phase.name,
                       gripper_joint=grip_rad)
            if terminated or truncated:
                break

        success = bool(info.get("success"))
        if success or args.keep_failures:
            path = rec.end_episode(success, extra={"seed": seed})
            kept += 1
            print(f"[{kept}/{args.episodes}] seed={seed} success={success} -> {path.name}")
        else:
            rec.end_episode(False)
            rec.episodes.pop()
            (args.out / f"episode_{len(rec.episodes):05d}.npz").unlink(missing_ok=True)
            print(f"[--] seed={seed} failed, discarded")

        if attempted > args.episodes * 8:
            print("Giving up: the expert is failing too often. Check the task config.")
            break

    meta = rec.write_meta()
    env.close()
    print(f"\nwrote {kept} episodes ({attempted} attempts) -> {meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
