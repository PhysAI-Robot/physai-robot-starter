"""Run a registered MuJoCo robot through the real ROS2 graph."""

from __future__ import annotations

import argparse
import _bootstrap  # noqa: F401


def main() -> int:
    import rclpy
    from physai.robots import available_robots, create_ros2_node

    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=available_robots(), default="so101")
    parser.add_argument("--config", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-ticks", type=int)
    parser.add_argument("--scenario", default=None)
    args = parser.parse_args()

    config = None
    if args.config:
        from physai.config import load_task_config

        config = load_task_config(args.config).env
    rclpy.init()
    node = rclpy.create_node(f"{args.robot}_mujoco_driver")
    driver = None
    try:
        node_kwargs = {"config": config}
        if args.scenario is not None:
            node_kwargs["scenario"] = args.scenario
        driver = create_ros2_node(args.robot, node, **node_kwargs)
        node.get_logger().info(f"{args.robot} ROS2 MuJoCo driver started")
        driver.run(seed=args.seed, max_ticks=args.max_ticks)
    finally:
        if driver is not None:
            driver.close()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())