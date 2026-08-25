from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("moai_jackal_spubert")
    default_config = f"{package_share}/config/real_jackal_social005.yaml"
    default_rviz_config = f"{package_share}/rviz/real_jackal_spubert.rviz"

    config_file = LaunchConfiguration("config_file")
    cmd_vel_topic = LaunchConfiguration("cmd_vel_topic")
    use_cuda = LaunchConfiguration("use_cuda")
    require_global_path = LaunchConfiguration("require_global_path")
    launch_rviz = LaunchConfiguration("launch_rviz")
    rviz_config = LaunchConfiguration("rviz_config")

    bridge = Node(
        package="moai_jackal_spubert",
        executable="real_jackal_spubert_bridge",
        name="real_jackal_spubert_bridge",
        output="screen",
        parameters=[
            config_file,
            {
                "use_cuda": use_cuda,
                "require_global_path": require_global_path,
            },
        ],
    )
    tracker = Node(
        package="moai_jackal_spubert",
        executable="safe_path_tracker",
        name="safe_path_tracker",
        output="screen",
        parameters=[config_file, {"cmd_vel_topic": cmd_vel_topic, "motion_enabled": False}],
    )
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="spubert_navigation_rviz",
        output="screen",
        arguments=["-d", rviz_config],
        condition=IfCondition(launch_rviz),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("config_file", default_value=default_config),
            DeclareLaunchArgument("cmd_vel_topic", default_value="/spu_bert/cmd_vel_dryrun"),
            DeclareLaunchArgument("use_cuda", default_value="true"),
            DeclareLaunchArgument("require_global_path", default_value="true"),
            DeclareLaunchArgument("launch_rviz", default_value="false"),
            DeclareLaunchArgument("rviz_config", default_value=default_rviz_config),
            bridge,
            tracker,
            rviz,
        ]
    )
