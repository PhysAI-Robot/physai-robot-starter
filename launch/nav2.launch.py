"""Launch a registered ROS2 simulation with a minimal Nav2 graph."""

from __future__ import annotations

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs" / "nav2" / "turtlebot4"


def generate_launch_description() -> LaunchDescription:
    robot = LaunchConfiguration("robot")
    params_file = LaunchConfiguration("params-file")
    map_file = LaunchConfiguration("map-file")
    max_ticks = LaunchConfiguration("max-ticks")
    nav2_nodes = ["controller_server", "planner_server", "behavior_server", "bt_navigator"]

    return LaunchDescription(
        [
            DeclareLaunchArgument("robot", default_value="turtlebot4"),
            DeclareLaunchArgument(
                "params-file", default_value=str(DEFAULT_CONFIG_DIR / "params.yaml")
            ),
            DeclareLaunchArgument(
                "map-file", default_value=str(DEFAULT_CONFIG_DIR / "map.yaml")
            ),
            DeclareLaunchArgument("max-ticks", default_value="5000"),
            ExecuteProcess(
                cmd=[
                    "uv", "run", "python", "scripts/run_ros2_sim.py",
                    "--robot", robot, "--max-ticks", max_ticks,
                ],
                cwd=str(REPO_ROOT),
                output="screen",
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="map_to_odom",
                arguments=["0", "0", "0", "0", "0", "0", "map", "odom"],
            ),
            Node(
                package="nav2_map_server",
                executable="map_server",
                name="map_server",
                parameters=[params_file, {"yaml_filename": map_file}],
                output="screen",
            ),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_map",
                parameters=[{"use_sim_time": True, "autostart": True, "node_names": ["map_server"]}],
                output="screen",
            ),
            Node(
                package="nav2_controller",
                executable="controller_server",
                name="controller_server",
                parameters=[params_file],
                remappings=[("cmd_vel", "/cmd_vel")],
                output="screen",
            ),
            Node(
                package="nav2_planner",
                executable="planner_server",
                name="planner_server",
                parameters=[params_file],
                output="screen",
            ),
            Node(
                package="nav2_behaviors",
                executable="behavior_server",
                name="behavior_server",
                parameters=[params_file],
                remappings=[("cmd_vel", "/cmd_vel")],
                output="screen",
            ),
            Node(
                package="nav2_bt_navigator",
                executable="bt_navigator",
                name="bt_navigator",
                parameters=[params_file],
                output="screen",
            ),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_navigation",
                parameters=[{"use_sim_time": True, "autostart": True, "node_names": nav2_nodes}],
                output="screen",
            ),
        ]
    )