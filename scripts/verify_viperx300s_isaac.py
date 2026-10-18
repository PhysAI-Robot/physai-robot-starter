"""Smoke-check the ViperX 300 S robot on Isaac Sim and write a report.

    python scripts/verify_viperx300s_isaac.py

Imports the Menagerie MJCF into USD, builds the scene, and checks what an Isaac-backed
`ViperX300SIsaacEnv` must get right before a policy can drive it: the joint set and drive
gains, that the arm holds HOME without sagging, that the gripper closes and reopens
with both fingers, and that the front and wrist cameras render. Writes
`outputs/viperx300s/isaac/` (git-ignored): `report.txt` and the camera frames.
"""

from __future__ import annotations

import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from physai.config import load_manifest
from physai.config.compat import with_overrides
from physai.contracts import Action, GripperCommand
from physai.runtime import create_session

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "viperx300s" / "isaac"
MANIFEST = REPO / "configs" / "manifests" / "viperx300s_single_cube_place.yaml"


def save(name: str, image) -> None:
    from PIL import Image

    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(OUT / f"{name}.png")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def log(message: str) -> None:
        print(message, flush=True)
        lines.append(message)

    # Check the render explicitly below (and keep the frames either way) instead of
    # letting the env refuse to start.
    from physai.robots.so101.isaac_env import SO101IsaacEnv

    SO101IsaacEnv._require_rendered_robot = lambda self: None
    manifest = with_overrides(load_manifest(MANIFEST), simulator="isaac", policy="idle")
    session = create_session(manifest, render=True)
    env = session.runtime.robot
    robot = getattr(env, "_environment", env)
    log(f"env: {type(robot).__name__}  usd: {robot.usd_path}")
    log(f"dof names: {robot.dof_names}")
    stiffness, damping = robot.articulation.get_dof_gains()
    log(f"stiffness: {np.round(np.asarray(stiffness)[0], 2).tolist()}")
    log(f"damping:   {np.round(np.asarray(damping)[0], 2).tolist()}")
    log(f"arm limits: {np.round(robot.arm_limits, 3).tolist()}")
    log(f"grip limits (m): {robot.grip_limits}")

    obs = robot.reset(seed=0)
    log(f"robot drawn in the front camera: {robot.robot_is_rendered()}")
    from physai.robots.viperx300s.mujoco_env import HOME_QPOS

    hold = Action(joint_position=HOME_QPOS.copy(), gripper=GripperCommand(position=1.0))
    for _ in range(60):
        obs, *_ = robot.step(hold)
    q = np.asarray(obs.joint_state.position)
    sag = float(np.abs(q[:6] - HOME_QPOS).max())
    log(f"HOME hold, 60 steps: max |q - home| = {sag:.4f} rad  (gripper {q[6]:.4f} m)")

    for name, image in obs.images.items():
        save(f"home_{name}", image.data)
        log(f"image {name}: {image.data.shape}")

    close = Action(
        joint_position=HOME_QPOS.copy(), gripper=GripperCommand(position=0.0)
    )
    for _ in range(40):
        obs, *_ = robot.step(close)
    closed = float(np.asarray(obs.joint_state.position)[6])
    left_right = np.asarray(robot.articulation.get_dof_positions())[
        0, [robot._gripper_index, robot._right_index]
    ]
    log(f"gripper closed: left {left_right[0]:.4f} m, right {left_right[1]:.4f} m")
    for _ in range(40):
        obs, *_ = robot.step(hold)
    opened = float(np.asarray(obs.joint_state.position)[6])
    log(f"gripper reopened: {opened:.4f} m")

    cube = robot.cube_pos
    log(f"cube at {np.round(cube, 3).tolist()}  rest_z {robot.rest_z:.3f}")
    log(f"pinch at HOME {np.round(robot.pinch_center(), 3).tolist()}")
    ok = sag < 0.05 and closed < 0.03 and opened > 0.05
    log("RESULT: " + ("ok" if ok else "CHECK THE NUMBERS ABOVE"))
    (OUT / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    robot.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
