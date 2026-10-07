"""Run a registered MuJoCo robot through the real ROS2 graph."""

from __future__ import annotations


from pathlib import Path

import _bootstrap  # noqa: F401
from _cli import new_parser
from _common_args import add_max_steps, add_robot, add_seed


def main() -> int:
    import rclpy

    from physai.robots import available_robots, create_ros2_node

    parser = new_parser(__doc__)
    add_robot(
        parser,
        choices=available_robots(),
        help="robot to drive (default: the manifest's, or so101)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="session manifest whose robot, scene and settings the node's "
        "environment uses (a one-robot manifest, for example "
        "configs/manifests/so101_single_cube_fixed_place.yaml)",
    )
    add_seed(parser)
    add_max_steps(
        parser, help="stop after this many steps (default: run until interrupted)"
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="scenario name passed to the robot node (default: the robot's own)",
    )
    args = parser.parse_args()

    config = None
    robot = args.robot
    if args.manifest:
        from physai.config import load_manifest
        from physai.runtime import robot_env_config

        manifest = load_manifest(args.manifest)
        if manifest.world is not None or len(manifest.robots) != 1:
            parser.error("--manifest must describe exactly one robot")
        robot = robot or manifest.robots[0].robot
        if robot != manifest.robots[0].robot:
            parser.error(
                f"--robot {robot!r} does not match the manifest's "
                f"{manifest.robots[0].robot!r}"
            )
        config = robot_env_config(manifest)
    robot = robot or "so101"
    rclpy.init()
    node = rclpy.create_node(f"{robot}_mujoco_driver")
    driver = None
    try:
        node_kwargs = {"config": config}
        if args.scenario is not None:
            node_kwargs["scenario"] = args.scenario
        driver = create_ros2_node(robot, node, **node_kwargs)
        node.get_logger().info(f"{robot} ROS2 MuJoCo driver started")
        driver.run(seed=args.seed, max_ticks=args.max_steps)
    finally:
        if driver is not None:
            driver.close()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
