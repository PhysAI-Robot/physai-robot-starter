"""Benchmark SO-101 FK/IK metrics over a deterministic reachable target set.

    uv run python scripts/benchmark_ik.py --targets 20 --json-out outputs/ik_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

import _bootstrap  # noqa: F401

from physai.robots.so101 import EnvConfig, SO101Env
from physai.robots.so101.kinematics import top_down_quat
from physai.sim import SceneConfig


TARGET_OFFSETS = (
    (0.00, -0.03, 0.01),
    (0.01, 0.04, 0.01),
    (-0.02, 0.07, 0.01),
    (0.02, 0.01, 0.015),
    (-0.01, -0.01, 0.02),
)


def benchmark(args: argparse.Namespace) -> dict:
    env = SO101Env(
        EnvConfig(
            scene=SceneConfig(camera_width=64, camera_height=64),
            render=False,
            max_steps=1,
        )
    )
    results: list[dict] = []
    try:
        observation = env.reset(seed=args.seed)
        q_init = observation.joint_state.position[:5]
        base_position = env.cube_pos.copy()
        target_quat = top_down_quat()
        offsets = [TARGET_OFFSETS[index % len(TARGET_OFFSETS)] for index in range(args.targets)]
        for index, offset in enumerate(offsets):
            target = base_position + np.asarray(offset, dtype=np.float64)
            started = time.perf_counter()
            result = env.kin.ik(
                target,
                q_init=q_init,
                target_quat_wxyz=target_quat,
            )
            runtime_ms = (time.perf_counter() - started) * 1000.0
            env.data.qpos[env.arm_qadr] = result.qpos
            mujoco.mj_forward(env.model, env.data)
            fk_started = time.perf_counter()
            pose = env.kin.fk(env.data)
            fk_runtime_ms = (time.perf_counter() - fk_started) * 1000.0
            jacobian_started = time.perf_counter()
            jacobian = env.kin.site_jacobian(env.data)
            jacobian_runtime_ms = (time.perf_counter() - jacobian_started) * 1000.0
            results.append(
                {
                    "index": index,
                    "target": target.tolist(),
                    "converged": bool(result.converged),
                    "position_error": result.position_error,
                    "orientation_error": result.orientation_error,
                    "iterations": result.iterations,
                    "runtime_ms": runtime_ms,
                    "fk_position_error": float(
                        np.linalg.norm(pose.pose.position.as_array() - target)
                    ),
                    "fk_runtime_ms": fk_runtime_ms,
                    "jacobian_shape": list(jacobian.shape),
                    "jacobian_runtime_ms": jacobian_runtime_ms,
                }
            )
    finally:
        env.close()

    position_errors = [item["position_error"] for item in results]
    orientation_errors = [item["orientation_error"] for item in results]
    iterations = [item["iterations"] for item in results]
    runtimes = [item["runtime_ms"] for item in results]
    fk_errors = [item["fk_position_error"] for item in results]
    fk_runtimes = [item["fk_runtime_ms"] for item in results]
    jacobian_runtimes = [item["jacobian_runtime_ms"] for item in results]
    successes = sum(item["converged"] for item in results)
    return {
        "seed": args.seed,
        "targets": len(results),
        "success_count": successes,
        "success_rate": successes / len(results),
        "mean_position_error": float(np.mean(position_errors)),
        "max_position_error": float(np.max(position_errors)),
        "mean_orientation_error": float(np.mean(orientation_errors)),
        "max_orientation_error": float(np.max(orientation_errors)),
        "mean_iterations": float(np.mean(iterations)),
        "max_iterations": int(np.max(iterations)),
        "mean_runtime_ms": float(np.mean(runtimes)),
        "max_runtime_ms": float(np.max(runtimes)),
        "mean_fk_position_error": float(np.mean(fk_errors)),
        "max_fk_position_error": float(np.max(fk_errors)),
        "mean_fk_runtime_ms": float(np.mean(fk_runtimes)),
        "mean_jacobian_runtime_ms": float(np.mean(jacobian_runtimes)),
        "jacobian_shape": results[0]["jacobian_shape"],
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.targets < 1:
        parser.error("--targets must be positive")

    report = benchmark(args)
    print(
        f"IK: success {report['success_count']}/{report['targets']} "
        f"= {report['success_rate']:.0%}, "
        f"mean_pos_error={report['mean_position_error']:.6f} m, "
        f"mean_rot_error={report['mean_orientation_error']:.6f} rad, "
        f"mean_iterations={report['mean_iterations']:.1f}, "
        f"mean_runtime={report['mean_runtime_ms']:.3f} ms"
    )
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"json -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
