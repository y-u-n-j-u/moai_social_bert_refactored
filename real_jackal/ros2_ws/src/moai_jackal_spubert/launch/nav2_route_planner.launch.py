from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("moai_jackal_spubert")
    default_params = f"{package_share}/config/nav2_route_planner.yaml"
    map_yaml = LaunchConfiguration("map")
    params_file = LaunchConfiguration("params_file")
    use_sim_time = LaunchConfiguration("use_sim_time")

    common_override = {"use_sim_time": use_sim_time}
    map_server = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[params_file, common_override, {"yaml_filename": map_yaml}],
    )
    amcl = Node(
        package="nav2_amcl",
        executable="amcl",
        name="amcl",
        output="screen",
        parameters=[params_file, common_override],
    )
    planner = Node(
        package="nav2_planner",
        executable="planner_server",
        name="planner_server",
        output="screen",
        parameters=[params_file, common_override],
    )
    lifecycle = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_route",
        output="screen",
        parameters=[
            params_file,
            {
                "use_sim_time": use_sim_time,
                "autostart": True,
                "node_names": ["map_server", "amcl", "planner_server"],
            },
        ],
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("map", description="Absolute path to the saved ROS map YAML"),
            DeclareLaunchArgument("params_file", default_value=default_params),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            map_server,
            amcl,
            planner,
            lifecycle,
        ]
    )
