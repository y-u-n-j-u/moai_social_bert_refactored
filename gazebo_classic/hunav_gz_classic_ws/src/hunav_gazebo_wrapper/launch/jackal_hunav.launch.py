import os
from os import environ, pathsep

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (IncludeLaunchDescription, DeclareLaunchArgument,
                            SetEnvironmentVariable, SetLaunchConfiguration)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch.substitutions import (LaunchConfiguration, PathJoinSubstitution,
                                   FindExecutable)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


ARGUMENTS = [
    DeclareLaunchArgument('x', default_value='0.0',
                          description='Robot initial X position'),
    DeclareLaunchArgument('y', default_value='0.0',
                          description='Robot initial Y position'),
    DeclareLaunchArgument('z', default_value='0.15',
                          description='Robot initial Z position'),
    DeclareLaunchArgument('yaw', default_value='0.0',
                          description='Robot initial Yaw'),
    DeclareLaunchArgument('world_name', default_value='',
                          description='Map yaml file path for localization'),
    DeclareLaunchArgument('navigation', default_value='True',
                          description='Launch Nav2 navigation stack'),
]


def generate_launch_description():
    ld = LaunchDescription(ARGUMENTS)
    declare_actions(ld)
    return ld


def declare_actions(launch_description: LaunchDescription):

    set_sim_time = SetLaunchConfiguration('use_sim_time', 'True')
    launch_description.add_action(set_sim_time)

    set_slam = SetLaunchConfiguration('slam', 'False')
    launch_description.add_action(set_slam)

    # ----------------------------------------------------------
    #   GAZEBO MODEL PATH
    # ----------------------------------------------------------
    pkg_path = get_package_prefix('jackal_description')
    rs_path = get_package_prefix('realsense2_description')
    model_path = os.path.join(pkg_path, 'share') + pathsep + os.path.join(rs_path, 'share')

    if 'GAZEBO_MODEL_PATH' in environ:
        model_path += pathsep + environ['GAZEBO_MODEL_PATH']

    launch_description.add_action(
        SetEnvironmentVariable('GAZEBO_MODEL_PATH', model_path)
    )
    launch_description.add_action(
        SetEnvironmentVariable('GAZEBO_RESOURCE_PATH', model_path)
    )

    # ----------------------------------------------------------
    #   GAZEBO PLUGIN PATH (prefer patched gazebo_ros2_control)
    # ----------------------------------------------------------
    plugin_prefix = get_package_prefix('gazebo_ros2_control')
    plugin_path = os.path.join(plugin_prefix, 'lib')
    if 'GAZEBO_PLUGIN_PATH' in environ:
        plugin_path += pathsep + environ['GAZEBO_PLUGIN_PATH']

    launch_description.add_action(
        SetEnvironmentVariable('GAZEBO_PLUGIN_PATH', plugin_path)
    )

    # ----------------------------------------------------------
    #   JACKAL SENSORS
    # ----------------------------------------------------------
    # The SPU-BERT pedestrian/Jackal data pipeline uses trajectories and map
    # crops, so Jackal lidar is disabled by default. Set HUNAV_JACKAL_LASER=1
    # only when testing a Nav2 stack that explicitly needs /front/scan.
    launch_description.add_action(SetEnvironmentVariable('JACKAL_LASER', environ.get('HUNAV_JACKAL_LASER', '0')))
    launch_description.add_action(SetEnvironmentVariable('JACKAL_LASER_MODEL', environ.get('HUNAV_JACKAL_LASER_MODEL', 'ust10')))
    launch_description.add_action(SetEnvironmentVariable('JACKAL_LASER_TOPIC', environ.get('HUNAV_JACKAL_LASER_TOPIC', 'front/scan')))
    launch_description.add_action(SetEnvironmentVariable('JACKAL_REALSENSE', environ.get('HUNAV_JACKAL_REALSENSE', '0')))

    wrapper = get_package_share_directory('hunav_gazebo_wrapper')
    launch_description.add_action(SetEnvironmentVariable(
        'JACKAL_URDF_EXTRAS',
        '/home/hunav_gz_classic_ws/src/hunav_gazebo_wrapper/launch/jackal_diff_drive.urdf.xacro'
    ))

    # ----------------------------------------------------------
    #   NAVIGATION (Nav2)
    # ----------------------------------------------------------
    config_jackal_velocity_controller = PathJoinSubstitution(
        [FindPackageShare('jackal_control'), 'config', 'control.yaml']
    )
    config_jackal_ekf = PathJoinSubstitution(
        [FindPackageShare('jackal_control'), 'config', 'localization.yaml']
    )
    config_imu_filter = PathJoinSubstitution(
        [FindPackageShare('jackal_control'), 'config', 'imu_filter.yaml']
    )

    nav_launch = PathJoinSubstitution([
        FindPackageShare('nav2_bringup'),
        'launch',
        'navigation_launch.py'
    ])
    nav2_bringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([nav_launch]),
        launch_arguments={
            'params_file': os.path.join(
                wrapper, 'launch', 'jackal_params', 'jackal_nav_public_sim.yaml'
            ),
            'use_sim_time': 'True',
        }.items(),
        condition=IfCondition(LaunchConfiguration('navigation'))
    )

    loc_launch = PathJoinSubstitution([
        FindPackageShare('nav2_bringup'),
        'launch',
        'localization_launch.py'
    ])
    loc_bringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([loc_launch]),
        launch_arguments={
            'params_file': os.path.join(
                wrapper, 'launch', 'jackal_params', 'jackal_nav_public_sim.yaml'
            ),
            'map': LaunchConfiguration('world_name'),
            'use_sim_time': 'True',
        }.items(),
        condition=IfCondition(LaunchConfiguration('navigation'))
    )

    slam_launch = PathJoinSubstitution([
        FindPackageShare('nav2_bringup'),
        'launch',
        'slam_launch.py'
    ])
    slam_bringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([slam_launch]),
        launch_arguments={
            'params_file': os.path.join(
                wrapper, 'launch', 'jackal_params', 'jackal_nav_public_sim.yaml'
            ),
            'use_sim_time': 'True',
        }.items(),
        condition=IfCondition(LaunchConfiguration('slam'))
    )

    rviz_launch = PathJoinSubstitution([
        FindPackageShare('nav2_bringup'),
        'launch',
        'rviz_launch.py'
    ])
    rviz_bringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([rviz_launch]),
        launch_arguments={
            'use_sim_time': 'True',
        }.items(),
        condition=IfCondition(LaunchConfiguration('navigation'))
    )

    launch_description.add_action(nav2_bringup_launch)
    launch_description.add_action(loc_bringup_launch)
    launch_description.add_action(slam_bringup_launch)
    launch_description.add_action(rviz_bringup_launch)

    # ----------------------------------------------------------
    #   ROBOT DESCRIPTION
    # ----------------------------------------------------------
    robot_description_command = [
        PathJoinSubstitution([FindExecutable(name='xacro')]),
        ' ',
        PathJoinSubstitution([
            FindPackageShare('jackal_description'), 'urdf', 'jackal.urdf.xacro'
        ]),
        ' ',
        'is_sim:=true',
        ' ',
        'gazebo_controllers:=',
        config_jackal_velocity_controller,
    ]

    launch_jackal_description = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('jackal_description'),
                'launch',
                'description.launch.py'
            ])
        ),
        launch_arguments=[('robot_description_command', robot_description_command)]
    )
    launch_description.add_action(launch_jackal_description)

    # ----------------------------------------------------------
    #   ROBOT SPAWN
    # ----------------------------------------------------------
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_jackal',
        arguments=[
            '-entity', 'jackal',
            '-topic', 'robot_description',
            '-x', LaunchConfiguration('x'),
            '-y', LaunchConfiguration('y'),
            '-z', LaunchConfiguration('z'),
            '-Y', LaunchConfiguration('yaw'),
            '-timeout', '120',
        ],
        output='screen',
    )
    launch_description.add_action(spawn_robot)

    # ----------------------------------------------------------
    #   ROBOT LOCALIZATION AND CONTROL
    # ----------------------------------------------------------
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_node',
        output='screen',
        parameters=[config_jackal_ekf, {'use_sim_time': True}],
    )
    imu_filter_node = Node(
        package='imu_filter_madgwick',
        executable='imu_filter_madgwick_node',
        name='imu_filter_node',
        output='screen',
        parameters=[config_imu_filter, {'use_sim_time': True}],
    )
    launch_description.add_action(ekf_node)
    launch_description.add_action(imu_filter_node)
