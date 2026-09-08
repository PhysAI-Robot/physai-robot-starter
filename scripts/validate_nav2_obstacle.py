"""Validate TurtleBot4 obstacle-aware navigation through the live ROS2 graph."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _wait_for_lifecycle(node_name: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["ros2", "lifecycle", "get", node_name],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and "active" in result.stdout.lower():
            return
        time.sleep(0.5)
    raise TimeoutError(f"{node_name} did not become active within {timeout:.0f}s")


def _wait_for_scan(timeout: float) -> float:
    import rclpy
    from sensor_msgs.msg import LaserScan

    context = rclpy.context.Context()
    rclpy.init(args=[], context=context)
    node = rclpy.create_node("validate_turtlebot4_obstacle_scan", context=context)
    executor = rclpy.executors.SingleThreadedExecutor(context=context)
    executor.add_node(node)
    scans: list[LaserScan] = []
    node.create_subscription(LaserScan, "/scan", scans.append, 10)
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
            if scans:
                finite_ranges = [value for value in scans[-1].ranges if value == value]
                if finite_ranges:
                    return min(finite_ranges)
    finally:
        executor.remove_node(node)
        node.destroy_node()
        if context.ok():
            rclpy.shutdown(context=context)
    raise TimeoutError(f"/scan did not publish within {timeout:.0f}s")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    parser.add_argument("--goal-timeout", type=float, default=150.0)
    parser.add_argument("--max-position-error", type=float, default=0.35)
    args = parser.parse_args()

    launch_command = [
        "ros2",
        "launch",
        "launch/nav2.launch.py",
        "robot:=turtlebot4",
        "scenario:=obstacle_course",
        f"map-file:={REPO_ROOT / 'configs/nav2/turtlebot4/obstacle_map.yaml'}",
    ]
    process = subprocess.Popen(
        launch_command,
        cwd=REPO_ROOT,
        start_new_session=True,
    )
    try:
        _wait_for_lifecycle("/bt_navigator", args.startup_timeout)
        _wait_for_lifecycle("/collision_monitor", args.startup_timeout)
        scan_min = _wait_for_scan(args.startup_timeout)
        print(f"Obstacle scan minimum: {scan_min:.3f} m")
        if scan_min >= 0.8:
            raise RuntimeError("/scan did not detect the configured physical obstacle")

        goal_command = [
            "uv",
            "run",
            "python",
            "scripts/send_nav_goal.py",
            "--robot",
            "turtlebot4",
            "--x",
            "1.0",
            "--y",
            "0.0",
            "--yaw",
            "0.0",
            "--max-position-error",
            str(args.max_position_error),
        ]
        result = subprocess.run(
            goal_command,
            cwd=REPO_ROOT,
            timeout=args.goal_timeout,
            check=False,
        )
        return result.returncode
    finally:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10.0)


if __name__ == "__main__":
    raise SystemExit(main())
