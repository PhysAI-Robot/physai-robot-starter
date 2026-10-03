"""Render the same scene in MuJoCo and Isaac Sim and compare what the cameras see.

    uv run python scripts/compare_cameras.py --dump mujoco --out outputs/cams_mujoco.npz
    OMNI_KIT_ACCEPT_EULA=YES uv run python scripts/compare_cameras.py --dump isaac --out outputs/cams_isaac.npz
    uv run python scripts/compare_cameras.py --compare outputs/cams_mujoco.npz outputs/cams_isaac.npz

Camera-based policies see these images, so a difference in lighting, floor, or
material is a perception gap between the simulators. Each simulator is dumped
in its own process (one `SimulationApp` per process); both settle the arm at
the same joint targets with the cube at the scene's default position, then
render the front and wrist cameras. `--compare` reports, per camera, the
colour statistics of the background and the cube, what `ColorBlobDetector`
finds, and the mean pixel difference outside the robot, and can write a
side-by-side image.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from physai.contracts import Action, GripperCommand

# A DESCEND-like pose over the cube: five arm joints (rad), gripper open.
ARM_QPOS = np.array([-0.52, -0.103, 0.29, 1.331, -0.532])
GRIPPER = 1.0
SETTLE_STEPS = 30
CAMERAS = ("front", "wrist")


def _teleport_arm(env, sim: str) -> None:
    """Put the arm at ARM_QPOS without driving it there, so it never sweeps the cube."""
    if sim == "mujoco":
        import mujoco

        env.data.qpos[env.arm_qadr] = ARM_QPOS
        env.data.qpos[env.grip_qadr] = env.gripper_to_joint(GRIPPER)
        env.data.ctrl[env.arm_act_ids] = ARM_QPOS
        env.data.ctrl[env.grip_act_id] = env.gripper_to_joint(GRIPPER)
        mujoco.mj_forward(env.model, env.data)
        return
    target = np.zeros((1, len(env.dof_names)), dtype=np.float32)
    target[0, env._arm_indices] = ARM_QPOS
    target[0, env._gripper_index] = env.gripper_to_joint(GRIPPER)
    env.articulation.set_dof_positions(target)
    env.articulation.set_dof_position_targets(target)


def dump(sim: str, out: Path) -> None:
    from physai.sim.mujoco import PickPlaceMinimalSceneConfig

    scene = PickPlaceMinimalSceneConfig()
    if sim == "mujoco":
        from physai.robots.so101 import EnvConfig, SO101Env
        from physai.robots.so101.scene import scene_defaults

        scene = PickPlaceMinimalSceneConfig(**scene_defaults())
        env = SO101Env(EnvConfig(scene=scene, randomize_cube=False, render=True))
    else:
        from physai.robots.so101.isaac_env import IsaacEnvConfig, SO101IsaacEnv

        env = SO101IsaacEnv(
            IsaacEnvConfig(
                scene=scene, randomize_cube=False, cameras=CAMERAS, render=True
            )
        )
    env.reset(seed=0)
    _teleport_arm(env, sim)
    action = Action(joint_position=ARM_QPOS, gripper=GripperCommand(position=GRIPPER))
    for _ in range(SETTLE_STEPS):
        env.step(action)
    arrays = {name: np.asarray(env.render_camera(name)) for name in CAMERAS}
    arrays["cube"] = np.asarray(env.cube_pos)
    arm_pixels = int(robot_mask(arrays["front"]).sum())
    print(f"{sim}: arm pixels in the front image: {arm_pixels}")
    if arm_pixels < 500:
        # Isaac now and then renders a frame without the robot; comparing
        # such a dump would measure that glitch, not the scene.
        raise SystemExit("the robot is missing from the front image; rerun the dump")
    arrays["qpos"] = np.asarray(env.observe().joint_state.position)
    env.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    print(
        f"{sim}: cube {np.round(arrays['cube'], 4)} qpos {np.round(arrays['qpos'], 3)}"
    )
    print(f"wrote {out}")


def robot_mask(image: np.ndarray) -> np.ndarray:
    """The yellow arm, found by hue so it does not depend on how bright it is."""
    r, g, b = (image[..., i].astype(float) for i in range(3))
    saturated = (r - b) / np.maximum(r, 1.0) >= 0.4
    yellow = (g - b) / np.maximum(r - b, 1.0) >= 0.5
    return (r >= 60) & saturated & yellow


def cube_mask(image: np.ndarray) -> np.ndarray:
    """Reddish pixels that are not the arm: the cube in any shading."""
    r, g, b = (image[..., i].astype(int) for i in range(3))
    return (r > g + 40) & (r > b + 30) & ~robot_mask(image)


def region_means(image: np.ndarray) -> dict[str, np.ndarray | None]:
    """Mean RGB of the cube, the upper third (background) and the lower half
    without the arm or the cube (table and floor)."""
    height = image.shape[0]
    arm, cube = robot_mask(image), cube_mask(image)
    rows = np.arange(height)[:, None] * np.ones(image.shape[:2], dtype=int)
    free = ~arm & ~cube
    regions = {
        "cube": cube,
        "background": free & (rows < height // 3),
        "table/floor": free & (rows >= height // 2),
    }
    return {
        name: image[mask].mean(axis=0) if mask.sum() >= 20 else None
        for name, mask in regions.items()
    }


def detection(image: np.ndarray) -> dict:
    from research.classical_control.so101_visual_servo import ColorBlobDetector

    feature = ColorBlobDetector().detect(image)
    if feature is None:
        return {"found": False}
    return {
        "found": True,
        "pixel": feature.pixel,
        "area": feature.area,
        "confidence": feature.confidence,
    }


def pixel_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute RGB difference over pixels that are not the arm in either."""
    keep = ~robot_mask(a) & ~robot_mask(b)
    return float(np.abs(a.astype(float) - b.astype(float))[keep].mean())


def _fmt(rgb: np.ndarray | None) -> str:
    return "-" if rgb is None else "(" + ", ".join(f"{v:.0f}" for v in rgb) + ")"


def compare(
    a_path: Path, b_path: Path, label_a: str, label_b: str, montage: Path | None
):
    a, b = np.load(a_path), np.load(b_path)
    print(
        f"cube {label_a} {np.round(a['cube'], 4)}  {label_b} {np.round(b['cube'], 4)}"
    )
    print(
        f"qpos {label_a} {np.round(a['qpos'], 3)}  {label_b} {np.round(b['qpos'], 3)}"
    )
    for name in CAMERAS:
        image_a, image_b = a[name], b[name]
        if image_a.shape != image_b.shape:
            raise SystemExit(
                f"{name}: image sizes differ {image_a.shape} {image_b.shape}"
            )
        print(f"\n== {name} camera ({image_a.shape[1]}x{image_a.shape[0]})")
        print(f"| region | {label_a} mean RGB | {label_b} mean RGB |")
        print("| --- | --- | --- |")
        means_a, means_b = region_means(image_a), region_means(image_b)
        for region in means_a:
            print(f"| {region} | {_fmt(means_a[region])} | {_fmt(means_b[region])} |")
        det_a, det_b = detection(image_a), detection(image_b)
        for label, det in ((label_a, det_a), (label_b, det_b)):
            if det["found"]:
                print(
                    f"detector {label}: pixel ({det['pixel'][0]:.1f}, {det['pixel'][1]:.1f})"
                    f" area {det['area']} confidence {det['confidence']:.2f}"
                )
            else:
                print(f"detector {label}: nothing found")
        if det_a["found"] and det_b["found"]:
            delta = det_b["pixel"] - det_a["pixel"]
            print(
                f"blob centroid {label_b} - {label_a}: ({delta[0]:+.1f}, {delta[1]:+.1f}) px"
            )
        print(
            f"mean abs pixel difference (outside the arm): {pixel_difference(image_a, image_b):.1f}"
        )
    if montage is not None:
        import imageio.v3 as iio

        rows = []
        for name in CAMERAS:
            diff = np.abs(a[name].astype(int) - b[name].astype(int)).astype(np.uint8)
            rows.append(np.concatenate([a[name], b[name], diff], axis=1))
        montage.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(montage, np.concatenate(rows, axis=0))
        print(f"\nmontage ({label_a} | {label_b} | abs difference) -> {montage}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dump", choices=("mujoco", "isaac"))
    mode.add_argument("--compare", nargs=2, type=Path, metavar=("A.npz", "B.npz"))
    parser.add_argument("--out", type=Path, help="with --dump: the .npz to write")
    parser.add_argument("--montage", type=Path, help="with --compare: write an image")
    parser.add_argument("--label-a", default="MuJoCo")
    parser.add_argument("--label-b", default="Isaac")
    args = parser.parse_args()
    if args.dump:
        if args.out is None:
            parser.error("--dump needs --out")
        dump(args.dump, args.out)
    else:
        compare(*args.compare, args.label_a, args.label_b, args.montage)
    return 0


if __name__ == "__main__":
    sys.exit(main())
