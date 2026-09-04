"""Run the SO-101 MuJoCo simulation through the real ROS2 graph."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401


def main() -> int:
    import rclpy
    from physai.bridge import SO101ROS2Node
    from physai.config import load_task_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-ticks", type=int)
    args = parser.parse_args()

    task_config = load_task_config(args.config) if args.config else None
    config = task_config.env if task_config else None
    if config is not None:
        config.seed = args.seed
    rclpy.init()
    node = rclpy.create_node("so101_mujoco_driver")
    driver = None
    try:
        driver = SO101ROS2Node(node, config=config)
        node.get_logger().info("SO-101 ROS2 MuJoCo driver started")
        driver.run(seed=args.seed, max_ticks=args.max_ticks)
    finally:
        if driver is not None:
            driver.close()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())