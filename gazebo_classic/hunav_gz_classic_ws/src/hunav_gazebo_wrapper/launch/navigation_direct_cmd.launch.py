#!/usr/bin/env python3
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterFile
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    """Launch Nav2 with one supervised velocity-command path.

    The stock Nav2 launch maps both controller_server and velocity_smoother
    through /cmd_vel_nav, while this project also relays supervised commands
    to /cmd_vel. Omitting velocity_smoother prevents two publishers from
    racing on /cmd_vel; PMB2's diff-drive controller retains acceleration and
    velocity limiting.
    """
    bringup_dir = get_package_share_directory("nav2_bringup")
    namespace = LaunchConfiguration("namespace")
    use_sim_time = LaunchConfiguration("use_sim_time")
    autostart = LaunchConfiguration("autostart")
    params_file = LaunchConfiguration("params_file")
    use_respawn = LaunchConfiguration("use_respawn")
    log_level = LaunchConfiguration("log_level")

    lifecycle_nodes = [
        "controller_server",
        "smoother_server",
        "planner_server",
        "behavior_server",
        "bt_navigator",
        "waypoint_follower",
    ]
    remappings = [("/tf", "tf"), ("/tf_static", "tf_static")]
    configured_params = ParameterFile(
        RewrittenYaml(
            source_file=params_file,
            root_key=namespace,
            param_rewrites={
                "use_sim_time": use_sim_time,
                "autostart": autostart,
            },
            convert_types=True,
        ),
        allow_substs=True,
    )

    common = {
        "output": "screen",
        "respawn": use_respawn,
        "respawn_delay": 2.0,
        "parameters": [configured_params],
        "arguments": ["--ros-args", "--log-level", log_level],
    }
    nodes = [
        Node(
            package="nav2_controller",
            executable="controller_server",
            remappings=remappings + [("cmd_vel", "cmd_vel_nav")],
            **common,
        ),
        Node(
            package="nav2_smoother",
            executable="smoother_server",
            name="smoother_server",
            remappings=remappings,
            **common,
        ),
        Node(
            package="nav2_planner",
            executable="planner_server",
            name="planner_server",
            remappings=remappings,
            **common,
        ),
        Node(
            package="nav2_behaviors",
            executable="behavior_server",
            name="behavior_server",
            remappings=remappings,
            **common,
        ),
        Node(
            package="nav2_bt_navigator",
            executable="bt_navigator",
            name="bt_navigator",
            remappings=remappings,
            **common,
        ),
        Node(
            package="nav2_waypoint_follower",
            executable="waypoint_follower",
            name="waypoint_follower",
            remappings=remappings,
            **common,
        ),
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_navigation",
            output="screen",
            arguments=["--ros-args", "--log-level", log_level],
            parameters=[
                {"use_sim_time": use_sim_time},
                {"autostart": autostart},
                {"node_names": lifecycle_nodes},
            ],
        ),
    ]

    description = LaunchDescription()
    description.add_action(
        SetEnvironmentVariable("RCUTILS_LOGGING_BUFFERED_STREAM", "1")
    )
    description.add_action(
        DeclareLaunchArgument("namespace", default_value="")
    )
    description.add_action(
        DeclareLaunchArgument("use_sim_time", default_value="false")
    )
    description.add_action(
        DeclareLaunchArgument(
            "params_file",
            default_value=os.path.join(
                bringup_dir,
                "params",
                "nav2_params.yaml",
            ),
        )
    )
    description.add_action(
        DeclareLaunchArgument("autostart", default_value="true")
    )
    description.add_action(
        DeclareLaunchArgument("use_respawn", default_value="False")
    )
    description.add_action(
        DeclareLaunchArgument("log_level", default_value="info")
    )
    for node in nodes:
        description.add_action(node)
    return description
