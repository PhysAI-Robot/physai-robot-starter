"""Regenerate the images and clips embedded in README.md.

    python scripts/render_docs_media.py                # everything
    python scripts/render_docs_media.py --only so101   # one group

Writes into `docs/media/`, which is committed so the README renders on GitHub
without anyone having to run the simulator first. Every clip uses a fixed seed,
so re-running this reproduces the committed files rather than merely similar
ones.

GIF rather than MP4 on purpose: GitHub renders an inline GIF in a README but
only links an MP4.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import imageio.v3 as iio
import numpy as np

from physai.contracts import Action, Twist, Vector3
from physai.planner import SortingPlanner
from physai.policy import create_policy
from physai.policy.plan_runner import PlanRunner
from physai.robots.so101 import EnvConfig, SO101Env
from physai.robots.turtlebot import TurtleBot4Config, TurtleBot4Env
from physai.sim import SceneConfig
from physai.tasks import TaskRuntime, create_task

MEDIA = Path(__file__).resolve().parents[1] / "docs" / "media"


def save_gif(frames: list[np.ndarray], path: Path, fps: int, stride: int = 3) -> None:
    """Half resolution and every `stride`-th frame, to keep the README light."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clip = np.stack(frames)[::stride][:, ::2, ::2, :]
    iio.imwrite(path, clip, duration=stride * (1000 / fps), loop=0)
    print(f"  {path.name}  {clip.shape[0]} frames  {path.stat().st_size / 1024:.0f} KB")


def _so101(scene: SceneConfig, task: str, seed: int, max_steps: int = 400):
    robot = SO101Env(EnvConfig(scene=scene, seed=seed, max_steps=max_steps, render=True))
    return robot, TaskRuntime(robot, create_task(task))


def render_pick_place(seed: int = 0) -> None:
    print("[so101] scripted pick-and-place")
    robot, env = _so101(SceneConfig(camera_width=640, camera_height=480), "pick_place", seed)
    obs = env.reset(seed=seed)
    policy = create_policy("scripted", env=env)
    policy.reset(obs)

    frames, info = [], {}
    for _ in range(400):
        frames.append(env.render_camera("front"))
        obs, _, terminated, truncated, info = env.step(policy.act(obs))
        if terminated or truncated or policy.done:
            break
    print(f"  success={bool(info.get('success'))} steps={env.step_count}")
    save_gif(frames, MEDIA / "so101_pick_place.gif", fps=int(robot.cfg.control_hz))
    env.close()


def render_camera_views(seed: int = 0) -> None:
    """Both observation cameras at the moment the jaws close on the cube."""
    print("[so101] observation cameras")
    robot, env = _so101(SceneConfig(camera_width=480, camera_height=480), "pick_place", seed)
    obs = env.reset(seed=seed)
    policy = create_policy("scripted", env=env)
    policy.reset(obs)

    for _ in range(400):
        obs, _, terminated, truncated, _ = env.step(policy.act(obs))
        if policy.phase.name == "CLOSE" or terminated or truncated or policy.done:
            break

    for name in ("front", "wrist"):
        path = MEDIA / f"so101_camera_{name}.png"
        iio.imwrite(path, env.render_camera(name))
        print(f"  {path.name}  {path.stat().st_size / 1024:.0f} KB")
    env.close()


def render_sorting(seed: int = 0, planner_seed: int = 1) -> None:
    print("[so101] sorting task")
    scene = SceneConfig(num_cubes=3, camera_width=640, camera_height=480)

    robot, env = _so101(scene, "sorting", seed)
    obs = env.reset(seed=seed)
    policy = create_policy("scripted", env=env)
    policy.reset(obs)
    frames, info = [], {}
    for _ in range(600):
        frames.append(env.render_camera("front"))
        obs, _, terminated, truncated, info = env.step(policy.act(obs))
        if terminated or truncated or policy.done:
            break
    print(f"  scripted target={env.target_color} success={bool(info.get('success'))}")
    save_gif(frames, MEDIA / "sorting" / "sorting_scripted.gif", fps=int(robot.cfg.control_hz))
    env.close()

    # Same layout, one word different in the instruction.
    for color in ("red", "blue"):
        robot, env = _so101(scene, "sorting", planner_seed, max_steps=800)
        obs = env.reset(seed=planner_seed)
        plan = SortingPlanner(env, env.target_pos).plan(
            f"put the {color} cube on the green pad", obs
        )
        runner = PlanRunner(env.kin, plan, dt=env.control_dt)
        runner.reset(obs)
        frames = []
        for _ in range(800):
            frames.append(env.render_camera("front"))
            obs, _, terminated, truncated, _ = env.step(runner.act(obs))
            runner.note_progress(
                env.kin.pinch_center(env.data),
                env.joint_to_gripper(obs.joint_state.position[5]),
            )
            if runner.done or terminated or truncated:
                break
        print(f"  planner instruction={color}")
        save_gif(frames, MEDIA / "sorting" / f"sorting_planner_{color}.gif",
                 fps=int(robot.cfg.control_hz))
        env.close()


def render_turtlebot(seed: int = 0) -> None:
    print("[turtlebot4] base velocity")
    env = TurtleBot4Env(TurtleBot4Config(seed=seed, max_steps=300, render=True))
    env.reset(seed=seed)

    # Constant forward speed plus a constant yaw rate, so the base drives a
    # visible arc instead of a straight line out of frame.
    action = Action(ee_twist=Twist(linear=Vector3(x=0.45), angular=Vector3(z=0.5)))
    frames = []
    for _ in range(240):
        frames.append(env.render_camera())
        env.step(action)
    save_gif(frames, MEDIA / "turtlebot4_drive.gif",
             fps=int(env.cfg.control_hz), stride=5)
    env.close()


GROUPS = {
    "so101": (render_pick_place, render_camera_views),
    "sorting": (render_sorting,),
    "turtlebot4": (render_turtlebot,),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(GROUPS),
                    help="render one group instead of all of them")
    args = ap.parse_args()

    names = [args.only] if args.only else list(GROUPS)
    for name in names:
        for render in GROUPS[name]:
            render()
    print(f"media -> {MEDIA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
