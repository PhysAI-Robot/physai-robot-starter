"""Launch the TurtleBot4 MuJoCo driver with a deterministic static Nav2 map."""

from __future__ import annotations

from pathlib import Path

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).resolve().parent


def generate_launch_description() -> LaunchDescription:
    params_file = CONFIG_DIR / "nav2_params.yaml"
    map_file = CONFIG_DIR / "dummy_map.yaml"
    nav2_nodes = ["controller_server", "planner_server", "behavior_server", "bt_navigator"]
    nav2_parameters = [str(params_file)]

    return LaunchDescription(
        [
            ExecuteProcess(
                cmd=[
                    "uv",
                    "run",
                    "python",
                    "scripts/run_ros2_sim.py",
                    "--robot",
                    "turtlebot4",
                    "--max-ticks",
                    "5000",
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
                parameters=[
                    str(params_file),
                    {"yaml_filename": str(map_file)},
                ],
                output="screen",
            ),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_map",
                parameters=[
                    {
                        "use_sim_time": True,
                        "autostart": True,
                        "node_names": ["map_server"],
                    }
                ],
                output="screen",
            ),
            Node(
                package="nav2_controller",
                executable="controller_server",
                name="controller_server",
                parameters=nav2_parameters,
                remappings=[("cmd_vel", "/cmd_vel")],
                output="screen",
            ),
            Node(
                package="nav2_planner",
                executable="planner_server",
                name="planner_server",
                parameters=nav2_parameters,
                output="screen",
            ),
            Node(
                package="nav2_behaviors",
                executable="behavior_server",
                name="behavior_server",
                parameters=nav2_parameters,
                remappings=[("cmd_vel", "/cmd_vel")],
                output="screen",
            ),
            Node(
                package="nav2_bt_navigator",
                executable="bt_navigator",
                name="bt_navigator",
                parameters=nav2_parameters,
                output="screen",
            ),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_navigation",
                parameters=[
                    {
                        "use_sim_time": True,
                        "autostart": True,
                        "node_names": nav2_nodes,
                    }
                ],
                output="screen",
            ),
        ]
    )