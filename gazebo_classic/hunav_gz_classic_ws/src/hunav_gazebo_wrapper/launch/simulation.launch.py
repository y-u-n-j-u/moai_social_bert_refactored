from os import path
from os import environ
from os import pathsep
from scripts import GazeboRosPaths
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from ament_index_python.packages import PackageNotFoundError

from launch import LaunchDescription
from launch.actions import (IncludeLaunchDescription, SetEnvironmentVariable, 
                            DeclareLaunchArgument, ExecuteProcess, Shutdown, 
                            RegisterEventHandler, TimerAction, LogInfo)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (PathJoinSubstitution, TextSubstitution,
                            LaunchConfiguration, PythonExpression, EnvironmentVariable)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from launch.event_handlers import (OnExecutionComplete, OnProcessExit,
                                OnProcessIO, OnProcessStart, OnShutdown)

def generate_launch_description():

    # Do not force NVIDIA GLX here. The docker runner hides NVIDIA devices,
    # so forcing the NVIDIA GLX vendor can make gzclient stop responding.
    use_nvidia_gpu = []

    # World generation parameters
    #environment_name = LaunchConfiguration('environment_name')
    gz_obs = LaunchConfiguration('use_gazebo_obs')
    rate = LaunchConfiguration('update_rate')
    robot_name = LaunchConfiguration('robot_name')
    global_frame = LaunchConfiguration('global_frame_to_publish')
    use_navgoal = LaunchConfiguration('use_navgoal_to_start')
    navgoal_topic = LaunchConfiguration('navgoal_topic')
    ignore_models = LaunchConfiguration('ignore_models')
    use_gazebo_gui = LaunchConfiguration('use_gazebo_gui')
    use_rviz = LaunchConfiguration('use_rviz')
    use_hunav_evaluator = LaunchConfiguration('use_hunav_evaluator')
    navigation = LaunchConfiguration('navigation')
    use_static_map_odom = LaunchConfiguration('use_static_map_odom')
    robot_type = LaunchConfiguration('robot_type')
    agent_motion_model = LaunchConfiguration('agent_motion_model')
    social_bert_predictor = LaunchConfiguration('social_bert_predictor')
    spubert_model_path = LaunchConfiguration('spubert_model_path')
    spubert_repo_path = LaunchConfiguration('spubert_repo_path')
    moai_spubert_repo_path = LaunchConfiguration('moai_spubert_repo_path')
    spubert_cuda = LaunchConfiguration('spubert_cuda')
    spubert_k_sample = LaunchConfiguration('spubert_k_sample')
    spubert_d_sample = LaunchConfiguration('spubert_d_sample')
    spubert_cache_ttl = LaunchConfiguration('spubert_cache_ttl')
    spubert_refresh_dt = LaunchConfiguration('spubert_refresh_dt')
    spubert_sync_on_cache_miss = LaunchConfiguration('spubert_sync_on_cache_miss')
    spubert_map_yaml_path = LaunchConfiguration('spubert_map_yaml_path')
    spubert_use_map_collision_filter = LaunchConfiguration('spubert_use_map_collision_filter')
    spubert_map_collision_radius = LaunchConfiguration('spubert_map_collision_radius')
    spubert_map_collision_weight = LaunchConfiguration('spubert_map_collision_weight')
    spubert_hard_map_collision_guard = LaunchConfiguration('spubert_hard_map_collision_guard')
    spubert_candidate_selection_mode = LaunchConfiguration('spubert_candidate_selection_mode')
    spubert_guidance_point_radius = LaunchConfiguration('spubert_guidance_point_radius')
    social_bert_lookahead_step = LaunchConfiguration('social_bert_lookahead_step')
    social_bert_min_goal_speed = LaunchConfiguration('social_bert_min_goal_speed')
    social_bert_max_speed = LaunchConfiguration('social_bert_max_speed')
    social_bert_max_accel = LaunchConfiguration('social_bert_max_accel')
    social_bert_goal_velocity_blend = LaunchConfiguration('social_bert_goal_velocity_blend')
    social_bert_goal_arrival_distance = LaunchConfiguration('social_bert_goal_arrival_distance')
    social_bert_goal_arrival_min_speed = LaunchConfiguration('social_bert_goal_arrival_min_speed')
    social_bert_robot_personal_space = LaunchConfiguration('social_bert_robot_personal_space')
    social_bert_robot_collision_buffer = LaunchConfiguration('social_bert_robot_collision_buffer')
    social_bert_robot_hard_collision_guard = LaunchConfiguration('social_bert_robot_hard_collision_guard')
    social_bert_agent_personal_space = LaunchConfiguration('social_bert_agent_personal_space')
    social_bert_agent_avoidance_gain = LaunchConfiguration('social_bert_agent_avoidance_gain')
    social_bert_obstacle_avoidance_distance = LaunchConfiguration('social_bert_obstacle_avoidance_distance')
    social_bert_obstacle_avoidance_gain = LaunchConfiguration('social_bert_obstacle_avoidance_gain')
    social_bert_obstacle_collision_buffer = LaunchConfiguration('social_bert_obstacle_collision_buffer')
    social_bert_obstacle_lateral_speed_ratio = LaunchConfiguration('social_bert_obstacle_lateral_speed_ratio')
    social_bert_velocity_smoothing_alpha = LaunchConfiguration('social_bert_velocity_smoothing_alpha')
    social_bert_collision_buffer = LaunchConfiguration('social_bert_collision_buffer')
    social_bert_max_lateral_speed_ratio = LaunchConfiguration('social_bert_max_lateral_speed_ratio')
    social_bert_max_yaw_rate = LaunchConfiguration('social_bert_max_yaw_rate')
    social_bert_controller_avoidance_enabled = LaunchConfiguration('social_bert_controller_avoidance_enabled')
    social_bert_save_training_pkl = LaunchConfiguration('social_bert_save_training_pkl')
    social_bert_training_pkl_path = LaunchConfiguration('social_bert_training_pkl_path')
    social_bert_training_record_dt = LaunchConfiguration('social_bert_training_record_dt')
    social_bert_training_sample_stride = LaunchConfiguration('social_bert_training_sample_stride')
    social_bert_training_flush_every = LaunchConfiguration('social_bert_training_flush_every')
    social_bert_training_max_samples = LaunchConfiguration('social_bert_training_max_samples')
    social_bert_publish_debug_markers = LaunchConfiguration('social_bert_publish_debug_markers')
    social_bert_debug_marker_publish_every = LaunchConfiguration('social_bert_debug_marker_publish_every')
    social_bert_debug_focus_agent_id = LaunchConfiguration('social_bert_debug_focus_agent_id')
    social_bert_debug_focus_agent_only = LaunchConfiguration('social_bert_debug_focus_agent_only')
    social_bert_debug_guidance_only = LaunchConfiguration('social_bert_debug_guidance_only')
    social_bert_debug_show_all_candidate_paths = LaunchConfiguration('social_bert_debug_show_all_candidate_paths')
    social_bert_debug_show_all_model_io = LaunchConfiguration('social_bert_debug_show_all_model_io')
    social_bert_debug_scene_patch_dir = LaunchConfiguration('social_bert_debug_scene_patch_dir')
    social_bert_debug_scene_patch_agent_id = LaunchConfiguration('social_bert_debug_scene_patch_agent_id')
    social_bert_debug_scene_patch_every = LaunchConfiguration('social_bert_debug_scene_patch_every')
    social_bert_debug_model_io_image_dir = LaunchConfiguration('social_bert_debug_model_io_image_dir')
    social_bert_debug_model_io_image_agent_id = LaunchConfiguration('social_bert_debug_model_io_image_agent_id')
    social_bert_debug_model_io_image_every = LaunchConfiguration('social_bert_debug_model_io_image_every')
    jackal_spubert_controller = LaunchConfiguration('jackal_spubert_controller')
    robot_path_planner = LaunchConfiguration('robot_path_planner')
    robot_spubert_repo_path = LaunchConfiguration('robot_spubert_repo_path')
    robot_spubert_config_path = LaunchConfiguration('robot_spubert_config_path')
    robot_spubert_checkpoint_path = LaunchConfiguration('robot_spubert_checkpoint_path')
    robot_spubert_cuda = LaunchConfiguration('robot_spubert_cuda')
    robot_spubert_d_sample = LaunchConfiguration('robot_spubert_d_sample')
    robot_spubert_replan_period = LaunchConfiguration('robot_spubert_replan_period')
    robot_spubert_fallback_to_nav2 = LaunchConfiguration('robot_spubert_fallback_to_nav2')
    robot_save_training_pkl = LaunchConfiguration('robot_save_training_pkl')
    robot_training_pkl_path = LaunchConfiguration('robot_training_pkl_path')
    robot_training_record_dt = LaunchConfiguration('robot_training_record_dt')
    robot_training_sample_stride = LaunchConfiguration('robot_training_sample_stride')
    robot_training_flush_every = LaunchConfiguration('robot_training_flush_every')
    robot_training_max_samples = LaunchConfiguration('robot_training_max_samples')
    auto_goal_enabled = LaunchConfiguration('auto_goal_enabled')
    auto_goal_mode = LaunchConfiguration('auto_goal_mode')
    auto_goal_waypoints = LaunchConfiguration('auto_goal_waypoints')
    auto_goal_seed = LaunchConfiguration('auto_goal_seed')
    auto_goal_min_distance = LaunchConfiguration('auto_goal_min_distance')
    auto_goal_max_distance = LaunchConfiguration('auto_goal_max_distance')
    auto_goal_clearance = LaunchConfiguration('auto_goal_clearance')
    auto_goal_min_episode_duration = LaunchConfiguration('auto_goal_min_episode_duration')
    auto_goal_timeout = LaunchConfiguration('auto_goal_timeout')
    auto_goal_max_goals = LaunchConfiguration('auto_goal_max_goals')
    learned_agent_motion = PythonExpression(
        ["'", agent_motion_model, "' in ['social_bert', 'spubert', 'moai_spubert']"]
    )

    # Robot parameters
    namespace = LaunchConfiguration('robot_namespace')
    scan_model = LaunchConfiguration('laser_model')
    use_rgbd = LaunchConfiguration('rgbd_sensors')
    gz_x = LaunchConfiguration('gzpose_x')
    gz_y = LaunchConfiguration('gzpose_y')
    gz_z = LaunchConfiguration('gzpose_z')
    gz_R = LaunchConfiguration('gzpose_R')
    gz_P = LaunchConfiguration('gzpose_P')
    gz_Y = LaunchConfiguration('gzpose_Y')


    # agent configuration file
    agent_conf_file = PathJoinSubstitution([
        FindPackageShare('hunav_gazebo_wrapper'),
        'scenarios',
        LaunchConfiguration('configuration_file')
    ])

    # Read the yaml file and load the parameters
    hunav_loader_node = Node(
        package='hunav_agent_manager',
        executable='hunav_loader',
        output='screen',
        parameters=[agent_conf_file]
        #arguments=['--ros-args', '--params-file', conf_file]
    )

    # world base file
    world_file = PathJoinSubstitution([
        FindPackageShare('hunav_gazebo_wrapper'),
        'worlds',
        PythonExpression(["'", LaunchConfiguration('environment_name'), ".world'"])
    ])

    # the node looks for the base_world file in the directory 'worlds'
    # of the package hunav_gazebo_plugin direclty. So we do not need to 
    # indicate the path
    hunav_gazebo_worldgen_node = Node(
        package='hunav_gazebo_wrapper',
        executable='hunav_gazebo_world_generator',
        output='screen',
        parameters=[{'base_world': world_file},
        {'use_gazebo_obs': gz_obs},
        {'update_rate': rate},
        {'robot_name': robot_name},
        {'global_frame_to_publish': global_frame},
        {'use_navgoal_to_start': use_navgoal},
        {'navgoal_topic': navgoal_topic},
        {'ignore_models': ignore_models}]
        #arguments=['--ros-args', '--params-file', conf_file]
    )

    ordered_launch_event = RegisterEventHandler(
        OnProcessStart(
            target_action=hunav_loader_node,
            on_start=[
                LogInfo(msg='HunNavLoader started, launching HuNav_Gazebo_world_generator after 2 seconds...'),
                TimerAction(
                    period=2.0,
                    actions=[hunav_gazebo_worldgen_node],
                )
            ]
        )
    )

    # Then, launch the generated world in Gazebo 
    my_gazebo_models = PathJoinSubstitution([
        FindPackageShare('hunav_gazebo_wrapper'),
        'models',
    ])
    pmb2_gazebo_models = path.join(get_package_prefix('pmb2_description'), 'share')
    optional_gazebo_model_paths = [pmb2_gazebo_models]
    try:
        optional_gazebo_model_paths.append(path.join(get_package_prefix('jackal_description'), 'share'))
    except PackageNotFoundError:
        pass

    config_file_name = 'params.yaml' 
    pkg_dir = get_package_share_directory('hunav_gazebo_wrapper') 
    config_file = path.join(pkg_dir, 'launch', config_file_name) 

    model, plugin, media = GazeboRosPaths.get_paths()
    #print('model:', model)

    if 'GAZEBO_MODEL_PATH' in environ:
        model += pathsep+environ['GAZEBO_MODEL_PATH']
    if 'GAZEBO_PLUGIN_PATH' in environ:
        plugin += pathsep+environ['GAZEBO_PLUGIN_PATH']
    if 'GAZEBO_RESOURCE_PATH' in environ:
        media += pathsep+environ['GAZEBO_RESOURCE_PATH']

    env = {
        'GAZEBO_MODEL_PATH': model,
        'GAZEBO_PLUGIN_PATH': plugin,
        'GAZEBO_RESOURCE_PATH': media
    }
    print('env:', env)

    gazebo_model_path_values = [
        EnvironmentVariable('GAZEBO_MODEL_PATH'),
        TextSubstitution(text=pathsep),
        my_gazebo_models,
    ]
    gazebo_resource_path_values = [
        EnvironmentVariable('GAZEBO_RESOURCE_PATH'),
        TextSubstitution(text=pathsep),
        my_gazebo_models,
    ]
    for model_path in optional_gazebo_model_paths:
        gazebo_model_path_values.extend([TextSubstitution(text=pathsep), model_path])
        gazebo_resource_path_values.extend([TextSubstitution(text=pathsep), model_path])

    set_env_gazebo_model = SetEnvironmentVariable(
        name='GAZEBO_MODEL_PATH',
        value=gazebo_model_path_values
    )
    set_env_gazebo_resource = SetEnvironmentVariable(
        name='GAZEBO_RESOURCE_PATH',
        value=gazebo_resource_path_values
    )
    set_env_gazebo_plugin = SetEnvironmentVariable(
        name='GAZEBO_PLUGIN_PATH', 
        value=[EnvironmentVariable('GAZEBO_PLUGIN_PATH'), plugin]
    )

    
    world_path = PathJoinSubstitution([
        FindPackageShare('hunav_gazebo_wrapper'),
        'worlds',
        'generatedWorld.world' #'empty_cafe.world' #'pmb2_cafe.world'
    ])

    gzserver_cmd = [
        use_nvidia_gpu,
        'gzserver ',
        #'--pause ',
        # Pass through arguments to gzserver
         world_path, 
        _boolean_command('verbose'), '',
        '-s ', 'libgazebo_ros_init.so',
        '-s ', 'libgazebo_ros_factory.so',
        #'-s ', #'libgazebo_ros_state.so',
        '--ros-args',
        '--params-file', config_file,
    ]

    gzclient_cmd = [
        use_nvidia_gpu,
        'gzclient',
        _boolean_command('verbose'), ' ',
    ]

    gzserver_process = ExecuteProcess(
        cmd=gzserver_cmd,
        output='screen',
        #additional_env=env,
        shell=True,
        on_exit=Shutdown(),
        #condition=IfCondition(LaunchConfiguration('server_required')),
    )

    gzclient_process = ExecuteProcess(
        cmd=gzclient_cmd,
        output='screen',
        #additional_env=env,
        shell=True,
        condition=IfCondition(use_gazebo_gui),
        #condition=IfCondition(LaunchConfiguration('server_required')),
    )

    

    pmb2_gazebo_launch = PathJoinSubstitution([
        FindPackageShare("hunav_gazebo_wrapper"),
        "launch",
        "pmb2_hunav.launch.py"
    ],)

    jackal_gazebo_launch = PathJoinSubstitution([
        FindPackageShare("hunav_gazebo_wrapper"),
        "launch",
        "jackal_hunav.launch.py"
    ],)
      
    map_path = PathJoinSubstitution([
        FindPackageShare("hunav_gazebo_wrapper"),
        "maps",
        PythonExpression(["'", LaunchConfiguration('environment_name'), ".yaml'"])
    ],)
    # map_dir = get_package_share_directory('hunav_gazebo_wrapper') 
    # map_ext = PathJoinSubstitution([
    #     TextSubstitution(text=''),
    #     environment_name,
    #     TextSubstitution(text='.yaml')
    # ])
    # map_path = path.join(map_dir, 'maps', map_ext) 

    # 수정한 부분 3 !!!!!!!!!!!!!!##################################################################################
    pmb2_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([pmb2_gazebo_launch]),
        launch_arguments={
            'robot_name':  robot_name,
            'laser_model':  scan_model,
            'add_on_module': 'no-add-on',
            'is_public_sim': 'True',
            'use_sim_time': 'True',
            'world_name': map_path,
            'x': gz_x,
            'y': gz_y,
            'yaw': gz_Y,
            'navigation': navigation,
            'use_rviz': use_rviz,
            'advanced_navigation': 'False',
            'slam': 'False',
        }.items(),
        condition=IfCondition(PythonExpression(["'", robot_type, "' == 'pmb2'"]))
    )

    jackal_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([jackal_gazebo_launch]),
        launch_arguments={
            'world_name': map_path,
            'x': gz_x,
            'y': gz_y,
            'z': gz_z,
            'yaw': gz_Y,
            'navigation': navigation,
        }.items(),
        condition=IfCondition(PythonExpression(["'", robot_type, "' == 'jackal'"]))
    )

    def launch_gazebo_when_world_is_ready(event):
        text = event.text.decode(errors='replace')
        if 'New world file created!' not in text:
            return []

        return [
            LogInfo(msg='Generated world is ready, launching Gazebo...'),
            gzserver_process,
            gzclient_process,
        ]

    gz_launch_event = RegisterEventHandler(
        OnProcessIO(
            target_action=hunav_gazebo_worldgen_node,
            on_stdout=launch_gazebo_when_world_is_ready,
            on_stderr=launch_gazebo_when_world_is_ready,
        )
    )

    robot_launch_event = RegisterEventHandler(
        OnProcessStart(
            target_action=gzserver_process,
            on_start=[
                LogInfo(msg=['Gazebo server started, launching robot_type=', robot_type, ' after 30 seconds...']),
                TimerAction(
                    period=30.0,
                    actions=[pmb2_gazebo, jackal_gazebo],
                )
            ]
        )
    )

    hunav_manager_node = Node(
        package='hunav_agent_manager',
        executable='hunav_agent_manager',
        name='hunav_agent_manager',
        output='screen',
        parameters=[{'use_sim_time': True}],
        condition=UnlessCondition(learned_agent_motion)
    )

    spubert_manager_node = Node(
        package='moai_hunav_bridge',
        executable='spubert_compute_agents_node',
        name='spubert_compute_agents',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'predictor_mode': social_bert_predictor},
            {'spubert_model_path': spubert_model_path},
            {'spubert_repo_path': spubert_repo_path},
            {'moai_spubert_repo_path': moai_spubert_repo_path},
            {'spubert_cuda': spubert_cuda},
            {'spubert_k_sample': spubert_k_sample},
            {'spubert_d_sample': spubert_d_sample},
            {'spubert_cache_ttl': spubert_cache_ttl},
            {'spubert_refresh_dt': spubert_refresh_dt},
            {'spubert_sync_on_cache_miss': spubert_sync_on_cache_miss},
            {'spubert_map_yaml_path': spubert_map_yaml_path},
            {'spubert_use_map_collision_filter': spubert_use_map_collision_filter},
            {'spubert_map_collision_radius': spubert_map_collision_radius},
            {'spubert_map_collision_weight': spubert_map_collision_weight},
            {'spubert_hard_map_collision_guard': spubert_hard_map_collision_guard},
            {'spubert_candidate_selection_mode': spubert_candidate_selection_mode},
            {'guidance_point_radius': spubert_guidance_point_radius},
            {'lookahead_step': social_bert_lookahead_step},
            {'min_goal_speed': social_bert_min_goal_speed},
            {'max_speed': social_bert_max_speed},
            {'max_accel': social_bert_max_accel},
            {'goal_velocity_blend': social_bert_goal_velocity_blend},
            {'goal_arrival_distance': social_bert_goal_arrival_distance},
            {'goal_arrival_min_speed': social_bert_goal_arrival_min_speed},
            {'robot_personal_space': social_bert_robot_personal_space},
            {'robot_collision_buffer': social_bert_robot_collision_buffer},
            {'robot_hard_collision_guard': social_bert_robot_hard_collision_guard},
            {'agent_personal_space': social_bert_agent_personal_space},
            {'agent_avoidance_gain': social_bert_agent_avoidance_gain},
            {'obstacle_avoidance_distance': social_bert_obstacle_avoidance_distance},
            {'obstacle_avoidance_gain': social_bert_obstacle_avoidance_gain},
            {'obstacle_collision_buffer': social_bert_obstacle_collision_buffer},
            {'obstacle_lateral_speed_ratio': social_bert_obstacle_lateral_speed_ratio},
            {'velocity_smoothing_alpha': social_bert_velocity_smoothing_alpha},
            {'collision_buffer': social_bert_collision_buffer},
            {'max_lateral_speed_ratio': social_bert_max_lateral_speed_ratio},
            {'max_yaw_rate': social_bert_max_yaw_rate},
            {'controller_avoidance_enabled': social_bert_controller_avoidance_enabled},
            {'save_training_pkl': social_bert_save_training_pkl},
            {'training_pkl_path': social_bert_training_pkl_path},
            {'training_record_dt': social_bert_training_record_dt},
            {'training_sample_stride': social_bert_training_sample_stride},
            {'training_flush_every': social_bert_training_flush_every},
            {'training_max_samples': social_bert_training_max_samples},
            {'publish_debug_markers': social_bert_publish_debug_markers},
            {'debug_marker_publish_every': social_bert_debug_marker_publish_every},
            {'debug_focus_agent_id': social_bert_debug_focus_agent_id},
            {'debug_focus_agent_only': social_bert_debug_focus_agent_only},
            {'debug_guidance_only': social_bert_debug_guidance_only},
            {'debug_show_all_candidate_paths': social_bert_debug_show_all_candidate_paths},
            {'debug_show_all_model_io': social_bert_debug_show_all_model_io},
            {'debug_scene_patch_dir': social_bert_debug_scene_patch_dir},
            {'debug_scene_patch_agent_id': social_bert_debug_scene_patch_agent_id},
            {'debug_scene_patch_every': social_bert_debug_scene_patch_every},
            {'debug_model_io_image_dir': social_bert_debug_model_io_image_dir},
            {'debug_model_io_image_agent_id': social_bert_debug_model_io_image_agent_id},
            {'debug_model_io_image_every': social_bert_debug_model_io_image_every},
            {'obs_len': 8},
            {'pred_len': 12},
            {'prediction_dt': 0.4},
        ],
        condition=IfCondition(learned_agent_motion)
    )

    human_obstacle_cloud_node = Node(
        package='moai_hunav_bridge',
        executable='human_obstacle_cloud_node',
        name='human_obstacle_cloud',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'human_states_topic': '/human_states'},
            {'predicted_paths_topic': '/moai/social_bert_predicted_paths'},
            {'cloud_topic': '/moai/human_obstacle_cloud'},
            {'publish_rate': 5.0},
            {'current_ring_points': 6},
            {'clearing_ring_points': 36},
        ],
        condition=IfCondition(PythonExpression([
            "'", navigation, "' == 'True' and '", robot_type,
            "' in ['jackal', 'pmb2']"
        ]))
    )

    # The PAL Humble velocity_smoother is active and subscribed to
    # /cmd_vel_nav, but it intermittently emits no /cmd_vel messages in this
    # Gazebo image. PMB2's diff-drive controller already enforces acceleration
    # limits, so relay Nav2 commands to the existing twist_mux input.
    pmb2_cmd_vel_passthrough_node = Node(
        package='moai_hunav_bridge',
        executable='cmd_vel_passthrough_node',
        name='cmd_vel_passthrough',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'input_topic': '/cmd_vel_nav'},
            {'output_topic': '/cmd_vel'},
        ],
        condition=IfCondition(PythonExpression([
            "'", navigation, "' == 'True' and '", robot_type,
            "' == 'pmb2' and '", robot_path_planner, "' == 'nav2'"
        ]))
    )

    spubert_nav2_bridge_node = Node(
        package='moai_hunav_bridge',
        executable='spubert_nav2_bridge_node',
        name='spubert_nav2_bridge',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'execution_mode': robot_path_planner},
            {'robot_topic': '/robot_states'},
            {'humans_topic': '/human_states'},
            {'goal_topic': '/goal_pose'},
            {'predicted_humans_topic': '/moai/social_bert_predicted_paths'},
            {'path_topic': '/moai/spubert_robot_path'},
            {'marker_topic': '/moai/spubert_robot_path_markers'},
            {'status_topic': '/moai/spubert_robot_planner_status'},
            {'model_repo_path': robot_spubert_repo_path},
            {'model_config_path': robot_spubert_config_path},
            {'model_checkpoint_path': robot_spubert_checkpoint_path},
            {'map_yaml_path': spubert_map_yaml_path},
            {'use_cuda': robot_spubert_cuda},
            {'d_sample': robot_spubert_d_sample},
            {'guidance_radius': spubert_guidance_point_radius},
            {'obs_len': 8},
            {'pred_len': 12},
            {'prediction_dt': 0.4},
            {'replan_period': robot_spubert_replan_period},
            {'goal_tolerance': 0.40},
            {'robot_radius': 0.275},
            {'static_safety_margin': 0.10},
            {'min_human_center_distance': 1.20},
            {'human_safety_margin': 0.25},
            {'max_robot_speed': 1.50},
            {'fallback_to_nav2': robot_spubert_fallback_to_nav2},
        ],
        condition=IfCondition(PythonExpression([
            "'", navigation, "' == 'True' and '", robot_type, "' == 'pmb2'"
        ]))
    )

    spubert_jackal_controller_node = Node(
        package='moai_hunav_bridge',
        executable='spubert_jackal_controller_node',
        name='spubert_jackal_controller',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'robot_topic': '/robot_states'},
            {'humans_topic': '/human_states'},
            {'goal_topic': '/goal_pose'},
            {'cmd_vel_topic': '/cmd_vel'},
            {'marker_topic': '/moai/spubert_jackal_paths'},
            {'predicted_humans_topic': '/moai/social_bert_predicted_paths'},
            {'spubert_model_path': spubert_model_path},
            {'spubert_repo_path': spubert_repo_path},
            {'spubert_cuda': spubert_cuda},
            {'spubert_k_sample': spubert_k_sample},
            {'spubert_d_sample': spubert_d_sample},
            {'lookahead_step': 3},
            {'pred_len': 12},
            {'prediction_dt': 0.4},
            {'rollout_linear_samples': 2},
            {'rollout_angular_samples': 9},
        ],
        condition=IfCondition(PythonExpression(["'", jackal_spubert_controller, "' == 'True' and '", robot_type, "' == 'jackal'"]))
    )

    robot_dataset_logger_node = Node(
        package='moai_hunav_bridge',
        executable='jackal_teleop_dataset_logger_node',
        name='robot_target_dataset_logger',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'enabled': robot_save_training_pkl},
            {'robot_topic': '/robot_states'},
            {'human_states_topic': '/human_states'},
            {'goal_topic': '/goal_pose'},
            {'output_path': robot_training_pkl_path},
            {'obs_len': 8},
            {'pred_len': 12},
            {'record_dt': robot_training_record_dt},
            {'guidance_point_radius': spubert_guidance_point_radius},
            {'goal_reached_tolerance': 0.6},
            {'episode_timeout': ParameterValue(auto_goal_timeout, value_type=float)},
            {'require_goal': True},
            {'sample_stride': robot_training_sample_stride},
            {'flush_every': robot_training_flush_every},
            {'max_samples': robot_training_max_samples},
            {'stale_timeout': 2.0},
        ],
        condition=IfCondition(PythonExpression([
            "'", robot_save_training_pkl, "' == 'True' and '", robot_type,
            "' in ['jackal', 'pmb2']"
        ]))
    )

    random_goal_publisher_node = Node(
        package='moai_hunav_bridge',
        executable='random_goal_publisher_node',
        name='random_goal_publisher',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'goal_topic': '/goal_pose'},
            {'map_topic': '/map'},
            {'robot_topic': '/robot_states'},
            {'map_frame': 'map'},
            {'goal_mode': auto_goal_mode},
            {'environment_name': LaunchConfiguration('environment_name')},
            {'waypoint_route': auto_goal_waypoints},
            {'seed': auto_goal_seed},
            {'min_goal_distance': ParameterValue(auto_goal_min_distance, value_type=float)},
            {'max_goal_distance': ParameterValue(auto_goal_max_distance, value_type=float)},
            {'clearance': ParameterValue(auto_goal_clearance, value_type=float)},
            {'min_episode_duration': ParameterValue(auto_goal_min_episode_duration, value_type=float)},
            {'goal_timeout': ParameterValue(auto_goal_timeout, value_type=float)},
            {'max_goals': auto_goal_max_goals},
        ],
        condition=IfCondition(PythonExpression([
            "'", auto_goal_enabled, "' == 'True' and '", navigation,
            "' == 'True' and '", robot_type, "' == 'pmb2'"
        ]))
    )

    metrics_file = PathJoinSubstitution([
        FindPackageShare('hunav_evaluator'),
        'config',
        LaunchConfiguration('metrics_file')
    ])
    # hunav_evaluator node
    hunav_evaluator_node = Node(
        package='hunav_evaluator',
        executable='hunav_evaluator_node',
        output='screen',
        parameters=[metrics_file],
        condition=IfCondition(use_hunav_evaluator)
    )

    static_tf_node = Node(
        package = "tf2_ros", 
        executable = "static_transform_publisher",
        output='screen',
        arguments = ['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        condition=IfCondition(use_static_map_odom)
    )

    ground_truth_localization_node = Node(
        package='moai_hunav_bridge',
        executable='ground_truth_localization_node',
        name='ground_truth_localization',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'truth_topic': '/ground_truth_odom'},
            {'wheel_odom_topic': '/mobile_base_controller/odom'},
            {'map_frame': 'map'},
            {'odom_frame': 'odom'},
        ],
        condition=IfCondition(PythonExpression([
            "'", navigation, "' == 'True' and '", robot_type,
            "' == 'pmb2' and '", use_static_map_odom, "' == 'False'"
        ]))
    )

    manager_launch_event = RegisterEventHandler(
        OnProcessStart(
            target_action=hunav_loader_node,
            on_start=[
                LogInfo(msg=['HunNavLoader started, launching agent motion model=', agent_motion_model, ' after 3 seconds...']),
                TimerAction(
                    period=3.0,
                    actions=[hunav_manager_node, spubert_manager_node],
                )
            ]
        )
    )

    declare_agents_conf_file = DeclareLaunchArgument(
        'configuration_file', default_value='agents_training_corridor_medium.yaml',
        description='Specify configuration file name in the cofig directory'
    )
    declare_metrics_conf_file = DeclareLaunchArgument(
        'metrics_file', default_value='metrics.yaml',
        description='Specify the name of the metrics configuration file in the cofig directory'
    )
    # declare_arg_world = DeclareLaunchArgument(
    #     'base_world', default_value='no_roof_small_warehouse.world',
    #     description='Specify world file name'
    # )
    declare_arg_environment = DeclareLaunchArgument(
        'environment_name', default_value='training_corridor',
        description='Specify the name of the environment. This is used to load the Gazebo world file and map file.'
    )

    declare_gz_obs = DeclareLaunchArgument(
        'use_gazebo_obs', default_value='True',
        description='Whether to fill the agents obstacles with closest Gazebo obstacle or not'
    )
    declare_update_rate = DeclareLaunchArgument(
        'update_rate', default_value=EnvironmentVariable('HUNAV_UPDATE_RATE', default_value='30.0'),
        description='Update rate of the plugin'
    )
    declare_robot_name = DeclareLaunchArgument(
        'robot_name', default_value='pmb2',
        description='Specify the name of the robot Gazebo model'
    )
    declare_robot_type = DeclareLaunchArgument(
        'robot_type', default_value='pmb2',
        description='Robot launch profile to use: none, pmb2, or jackal. If jackal, also set robot_name:=jackal.'
    )
    declare_use_gazebo_gui = DeclareLaunchArgument(
        'use_gazebo_gui',
        default_value=EnvironmentVariable('HUNAV_USE_GAZEBO_GUI', default_value='true'),
        description='Launch gzclient. Set false for headless/faster simulation.'
    )
    declare_use_rviz = DeclareLaunchArgument(
        'use_rviz',
        default_value=EnvironmentVariable('HUNAV_USE_RVIZ', default_value='True'),
        description='Launch RViz for PMB2 navigation.'
    )
    declare_use_hunav_evaluator = DeclareLaunchArgument(
        'use_hunav_evaluator',
        default_value=EnvironmentVariable('HUNAV_USE_EVALUATOR', default_value='true'),
        description='Launch hunav_evaluator_node.'
    )
    declare_agent_motion_model = DeclareLaunchArgument(
        'agent_motion_model',
        default_value=EnvironmentVariable('HUNAV_AGENT_MOTION_MODEL', default_value='hunav'),
        description='Agent motion model: hunav uses HuNavSim social force/BT; spubert uses the MOAI predictor bridge. social_bert is kept as a legacy alias.'
    )
    declare_social_bert_predictor = DeclareLaunchArgument(
        'social_bert_predictor',
        default_value=EnvironmentVariable('HUNAV_SOCIAL_BERT_PREDICTOR', default_value='constant_velocity'),
        description='Predictor used by the MOAI bridge: constant_velocity, spubert, or moai_spubert.'
    )
    declare_spubert_model_path = DeclareLaunchArgument(
        'spubert_model_path',
        default_value=EnvironmentVariable('HUNAV_SPUBERT_MODEL_PATH', default_value=''),
        description='Path to a SPU-BERT .pth checkpoint.'
    )
    declare_spubert_repo_path = DeclareLaunchArgument(
        'spubert_repo_path',
        default_value=EnvironmentVariable(
            'HUNAV_SPUBERT_REPO_PATH',
            default_value='/home/hunav_gz_classic_ws/src/moai_social_bert_refactored/runtime/SPUBERT',
        ),
        description='Path to the SPU-BERT runtime repository.'
    )
    declare_moai_spubert_repo_path = DeclareLaunchArgument(
        'moai_spubert_repo_path',
        default_value=EnvironmentVariable(
            'HUNAV_MOAI_SPUBERT_REPO_PATH',
            default_value='/home/hunav_gz_classic_ws/src/moai_social_bert_refactored'
        ),
        description='Path to the locally trained moai_social_bert_refactored repository.'
    )
    declare_spubert_cuda = DeclareLaunchArgument(
        'spubert_cuda',
        default_value=EnvironmentVariable('HUNAV_SPUBERT_CUDA', default_value='false'),
        description='Use CUDA for SPU-BERT inference when a GPU is visible.'
    )
    declare_spubert_k_sample = DeclareLaunchArgument(
        'spubert_k_sample',
        default_value=EnvironmentVariable('HUNAV_SPUBERT_K_SAMPLE', default_value='20'),
        description='Number of final SPU-BERT goal/trajectory hypotheses K.'
    )
    declare_spubert_d_sample = DeclareLaunchArgument(
        'spubert_d_sample',
        default_value=EnvironmentVariable('HUNAV_SPUBERT_D_SAMPLE', default_value='200'),
        description='Number of goal samples used by SPU-BERT inference.'
    )
    declare_spubert_cache_ttl = DeclareLaunchArgument(
        'spubert_cache_ttl',
        default_value=EnvironmentVariable('HUNAV_SPUBERT_CACHE_TTL', default_value='3.0'),
        description='Seconds to reuse the latest asynchronous SPU-BERT prediction.'
    )
    declare_spubert_refresh_dt = DeclareLaunchArgument(
        'spubert_refresh_dt',
        default_value=EnvironmentVariable('HUNAV_SPUBERT_REFRESH_DT', default_value='0.4'),
        description='Seconds between fresh SPU-BERT inferences per agent.'
    )
    declare_spubert_map_yaml_path = DeclareLaunchArgument(
        'spubert_map_yaml_path',
        default_value=PathJoinSubstitution([
            FindPackageShare('hunav_gazebo_wrapper'),
            'maps',
            PythonExpression(["'", LaunchConfiguration('environment_name'), ".yaml'"])
        ]),
        description='ROS map YAML used for local occupancy crops and SPU-BERT candidate collision filtering.'
    )
    declare_spubert_use_map_collision_filter = DeclareLaunchArgument(
        'spubert_use_map_collision_filter',
        default_value=EnvironmentVariable('HUNAV_SPUBERT_USE_MAP_COLLISION_FILTER', default_value='True'),
        description='If true, penalize SPU-BERT candidate paths that collide with the occupancy map.'
    )
    declare_spubert_map_collision_radius = DeclareLaunchArgument(
        'spubert_map_collision_radius',
        default_value=EnvironmentVariable('HUNAV_SPUBERT_MAP_COLLISION_RADIUS', default_value='0.35'),
        description='Pedestrian radius used when checking SPU-BERT candidates against the occupancy map.'
    )
    declare_spubert_map_collision_weight = DeclareLaunchArgument(
        'spubert_map_collision_weight',
        default_value=EnvironmentVariable('HUNAV_SPUBERT_MAP_COLLISION_WEIGHT', default_value='100.0'),
        description='Penalty added per colliding waypoint when selecting a SPU-BERT candidate path.'
    )
    declare_spubert_hard_map_collision_guard = DeclareLaunchArgument(
        'spubert_hard_map_collision_guard',
        default_value=EnvironmentVariable('HUNAV_SPUBERT_HARD_MAP_COLLISION_GUARD', default_value='false'),
        description='Block the next pedestrian step if it would enter an occupied map cell.'
    )
    declare_spubert_candidate_selection_mode = DeclareLaunchArgument(
        'spubert_candidate_selection_mode',
        default_value=EnvironmentVariable('HUNAV_SPUBERT_CANDIDATE_SELECTION_MODE', default_value='guidance_point'),
        description='SPU-BERT candidate selector: guidance_point, final_goal, or none.'
    )
    declare_spubert_guidance_point_radius = DeclareLaunchArgument(
        'spubert_guidance_point_radius',
        default_value=EnvironmentVariable('HUNAV_SPUBERT_GUIDANCE_RADIUS', default_value='8.0'),
        description='Radius in meters for current-centered pedestrian guidance point.'
    )
    declare_spubert_sync_on_cache_miss = DeclareLaunchArgument(
        'spubert_sync_on_cache_miss',
        default_value=EnvironmentVariable('HUNAV_SPUBERT_SYNC_ON_CACHE_MISS', default_value='True'),
        description='Run one blocking SPU-BERT inference when no cached candidate paths are available.'
    )
    declare_social_bert_lookahead_step = DeclareLaunchArgument(
        'social_bert_lookahead_step',
        default_value=EnvironmentVariable('HUNAV_SOCIAL_BERT_LOOKAHEAD_STEP', default_value='6'),
        description='Predicted trajectory point used as the local control target.'
    )
    declare_social_bert_min_goal_speed = DeclareLaunchArgument(
        'social_bert_min_goal_speed',
        default_value=EnvironmentVariable('HUNAV_SOCIAL_BERT_MIN_GOAL_SPEED', default_value='0.9'),
        description='Minimum goal-directed speed used when neural predictions are near-static.'
    )
    declare_social_bert_max_speed = DeclareLaunchArgument(
        'social_bert_max_speed',
        default_value=EnvironmentVariable('HUNAV_SOCIAL_BERT_MAX_SPEED', default_value='1.8'),
        description='Upper speed clamp for simulated pedestrians.'
    )
    declare_social_bert_max_accel = DeclareLaunchArgument(
        'social_bert_max_accel',
        default_value=EnvironmentVariable('HUNAV_SOCIAL_BERT_MAX_ACCEL', default_value='1.8'),
        description='Acceleration clamp for smoothing simulated pedestrian velocity commands.'
    )
    declare_social_bert_goal_velocity_blend = DeclareLaunchArgument(
        'social_bert_goal_velocity_blend',
        default_value=EnvironmentVariable('HUNAV_SOCIAL_BERT_GOAL_VELOCITY_BLEND', default_value='0.7'),
        description='Blend ratio from SPU-BERT velocity toward goal-following velocity.'
    )
    declare_social_bert_goal_arrival_distance = DeclareLaunchArgument(
        'social_bert_goal_arrival_distance',
        default_value=EnvironmentVariable('HUNAV_SOCIAL_BERT_GOAL_ARRIVAL_DISTANCE', default_value='1.5'),
        description='Distance from the active pedestrian goal where arrival slowdown takes over.'
    )
    declare_social_bert_goal_arrival_min_speed = DeclareLaunchArgument(
        'social_bert_goal_arrival_min_speed',
        default_value=EnvironmentVariable('HUNAV_SOCIAL_BERT_GOAL_ARRIVAL_MIN_SPEED', default_value='0.08'),
        description='Minimum commanded speed while approaching the active pedestrian goal.'
    )
    declare_social_bert_robot_personal_space = DeclareLaunchArgument(
        'social_bert_robot_personal_space',
        default_value=EnvironmentVariable('HUNAV_SOCIAL_BERT_ROBOT_PERSONAL_SPACE', default_value='1.25'),
        description='Clearance distance at which simulated pedestrians start avoiding the robot.'
    )
    declare_social_bert_robot_collision_buffer = DeclareLaunchArgument(
        'social_bert_robot_collision_buffer',
        default_value=EnvironmentVariable('HUNAV_SOCIAL_BERT_ROBOT_COLLISION_BUFFER', default_value='0.35'),
        description='Hard robot-pedestrian collision guard buffer added to robot and pedestrian radii.'
    )
    declare_social_bert_robot_hard_collision_guard = DeclareLaunchArgument(
        'social_bert_robot_hard_collision_guard',
        default_value=EnvironmentVariable('HUNAV_SOCIAL_BERT_ROBOT_HARD_COLLISION_GUARD', default_value='True'),
        description='Prevent pedestrian next-step positions from entering the robot collision buffer.'
    )
    declare_social_bert_agent_personal_space = DeclareLaunchArgument(
        'social_bert_agent_personal_space',
        default_value=EnvironmentVariable('HUNAV_SOCIAL_BERT_AGENT_PERSONAL_SPACE', default_value='1.25'),
        description='Clearance distance at which simulated pedestrians start avoiding each other.'
    )
    declare_social_bert_agent_avoidance_gain = DeclareLaunchArgument(
        'social_bert_agent_avoidance_gain',
        default_value=EnvironmentVariable('HUNAV_SOCIAL_BERT_AGENT_AVOIDANCE_GAIN', default_value='0.9'),
        description='Strength of pedestrian-pedestrian velocity repulsion.'
    )
    declare_social_bert_obstacle_avoidance_distance = DeclareLaunchArgument(
        'social_bert_obstacle_avoidance_distance',
        default_value=EnvironmentVariable('HUNAV_SOCIAL_BERT_OBSTACLE_AVOIDANCE_DISTANCE', default_value='1.35'),
        description='Distance at which pedestrians start avoiding closest HuNav obstacles.'
    )
    declare_social_bert_obstacle_avoidance_gain = DeclareLaunchArgument(
        'social_bert_obstacle_avoidance_gain',
        default_value=EnvironmentVariable('HUNAV_SOCIAL_BERT_OBSTACLE_AVOIDANCE_GAIN', default_value='0.45'),
        description='Strength of obstacle velocity repulsion.'
    )
    declare_social_bert_obstacle_collision_buffer = DeclareLaunchArgument(
        'social_bert_obstacle_collision_buffer',
        default_value=EnvironmentVariable('HUNAV_SOCIAL_BERT_OBSTACLE_COLLISION_BUFFER', default_value='0.42'),
        description='Hard pedestrian-obstacle clearance used by the collision resolver.'
    )
    declare_social_bert_obstacle_lateral_speed_ratio = DeclareLaunchArgument(
        'social_bert_obstacle_lateral_speed_ratio',
        default_value=EnvironmentVariable('HUNAV_SOCIAL_BERT_OBSTACLE_LATERAL_SPEED_RATIO', default_value='0.75'),
        description='Temporary lateral speed allowance when pedestrians are close to obstacles.'
    )
    declare_social_bert_velocity_smoothing_alpha = DeclareLaunchArgument(
        'social_bert_velocity_smoothing_alpha',
        default_value=EnvironmentVariable('HUNAV_SOCIAL_BERT_VELOCITY_SMOOTHING_ALPHA', default_value='0.35'),
        description='Low-pass smoothing factor for pedestrian velocity commands.'
    )
    declare_social_bert_collision_buffer = DeclareLaunchArgument(
        'social_bert_collision_buffer',
        default_value=EnvironmentVariable('HUNAV_SOCIAL_BERT_COLLISION_BUFFER', default_value='0.16'),
        description='Extra pedestrian-pedestrian clearance used by the collision resolver.'
    )
    declare_social_bert_max_lateral_speed_ratio = DeclareLaunchArgument(
        'social_bert_max_lateral_speed_ratio',
        default_value=EnvironmentVariable('HUNAV_SOCIAL_BERT_MAX_LATERAL_SPEED_RATIO', default_value='0.25'),
        description='Maximum lateral pedestrian speed as a ratio of desired speed.'
    )
    declare_social_bert_max_yaw_rate = DeclareLaunchArgument(
        'social_bert_max_yaw_rate',
        default_value=EnvironmentVariable('HUNAV_SOCIAL_BERT_MAX_YAW_RATE', default_value='1.0'),
        description='Maximum pedestrian yaw rate used to keep actor animation aligned with motion.'
    )
    declare_social_bert_controller_avoidance_enabled = DeclareLaunchArgument(
        'social_bert_controller_avoidance_enabled',
        default_value=EnvironmentVariable('HUNAV_SOCIAL_BERT_CONTROLLER_AVOIDANCE_ENABLED', default_value='False'),
        description='If true, apply extra controller-level social/obstacle repulsion after the selected neural trajectory.'
    )
    declare_social_bert_save_training_pkl = DeclareLaunchArgument(
        'social_bert_save_training_pkl',
        default_value=EnvironmentVariable('HUNAV_SOCIAL_BERT_SAVE_TRAINING_PKL', default_value='False'),
        description='Record HuNavSim trajectories as moai all_trajs training pkl samples.'
    )
    declare_social_bert_training_pkl_path = DeclareLaunchArgument(
        'social_bert_training_pkl_path',
        default_value=EnvironmentVariable(
            'HUNAV_SOCIAL_BERT_TRAINING_PKL_PATH',
            default_value='/tmp/moai_gazebo_all_trajs.pkl'
        ),
        description='Output path for recorded moai all_trajs pkl.'
    )
    declare_social_bert_training_record_dt = DeclareLaunchArgument(
        'social_bert_training_record_dt',
        default_value=EnvironmentVariable('HUNAV_SOCIAL_BERT_TRAINING_RECORD_DT', default_value='0.4'),
        description='Downsample interval in seconds for recorded trajectory frames.'
    )
    declare_social_bert_training_sample_stride = DeclareLaunchArgument(
        'social_bert_training_sample_stride',
        default_value=EnvironmentVariable('HUNAV_SOCIAL_BERT_TRAINING_SAMPLE_STRIDE', default_value='1'),
        description='Sliding-window stride in recorded frames for training samples.'
    )
    declare_social_bert_training_flush_every = DeclareLaunchArgument(
        'social_bert_training_flush_every',
        default_value=EnvironmentVariable('HUNAV_SOCIAL_BERT_TRAINING_FLUSH_EVERY', default_value='50'),
        description='Write the training pkl after this many new samples.'
    )
    declare_social_bert_training_max_samples = DeclareLaunchArgument(
        'social_bert_training_max_samples',
        default_value=EnvironmentVariable('HUNAV_SOCIAL_BERT_TRAINING_MAX_SAMPLES', default_value='0'),
        description='Maximum samples to keep in the recorded pkl; 0 means unlimited.'
    )
    declare_social_bert_publish_debug_markers = DeclareLaunchArgument(
        'social_bert_publish_debug_markers',
        default_value=EnvironmentVariable('HUNAV_SOCIAL_BERT_PUBLISH_DEBUG_MARKERS', default_value='true'),
        description='Publish SPU-BERT/Social-BERT debug MarkerArray messages.'
    )
    declare_social_bert_debug_marker_publish_every = DeclareLaunchArgument(
        'social_bert_debug_marker_publish_every',
        default_value=EnvironmentVariable('HUNAV_SOCIAL_BERT_DEBUG_MARKER_PUBLISH_EVERY', default_value='1'),
        description='Publish debug MarkerArray messages every N compute cycles.'
    )
    declare_social_bert_debug_focus_agent_id = DeclareLaunchArgument(
        'social_bert_debug_focus_agent_id',
        default_value=EnvironmentVariable('HUNAV_SOCIAL_BERT_DEBUG_FOCUS_AGENT_ID', default_value='1'),
        description='Agent id whose model input/output and candidate paths are expanded in RViz; <=0 shows all.'
    )
    declare_social_bert_debug_focus_agent_only = DeclareLaunchArgument(
        'social_bert_debug_focus_agent_only',
        default_value=EnvironmentVariable('HUNAV_SOCIAL_BERT_DEBUG_FOCUS_AGENT_ONLY', default_value='False'),
        description='If true, publish debug markers only for the focus agent.'
    )
    declare_social_bert_debug_guidance_only = DeclareLaunchArgument(
        'social_bert_debug_guidance_only',
        default_value=EnvironmentVariable('HUNAV_SOCIAL_BERT_DEBUG_GUIDANCE_ONLY', default_value='False'),
        description='If true, publish only pedestrian guidance-point construction markers.'
    )
    declare_social_bert_debug_show_all_candidate_paths = DeclareLaunchArgument(
        'social_bert_debug_show_all_candidate_paths',
        default_value=EnvironmentVariable('HUNAV_SOCIAL_BERT_DEBUG_SHOW_ALL_CANDIDATE_PATHS', default_value='False'),
        description='Show all SPU-BERT candidate path bundles instead of only the focus agent.'
    )
    declare_social_bert_debug_show_all_model_io = DeclareLaunchArgument(
        'social_bert_debug_show_all_model_io',
        default_value=EnvironmentVariable('HUNAV_SOCIAL_BERT_DEBUG_SHOW_ALL_MODEL_IO', default_value='False'),
        description='Show obs/pred numbered model input-output markers for every agent instead of only the focus agent.'
    )
    declare_social_bert_debug_scene_patch_dir = DeclareLaunchArgument(
        'social_bert_debug_scene_patch_dir',
        default_value=EnvironmentVariable('HUNAV_SOCIAL_BERT_DEBUG_SCENE_PATCH_DIR', default_value=''),
        description='If non-empty, save SPU-BERT local occupancy crop patch-grid images to this directory.'
    )
    declare_social_bert_debug_scene_patch_agent_id = DeclareLaunchArgument(
        'social_bert_debug_scene_patch_agent_id',
        default_value=EnvironmentVariable('HUNAV_SOCIAL_BERT_DEBUG_SCENE_PATCH_AGENT_ID', default_value='1'),
        description='Agent id whose local map crop patch image is saved; <=0 saves all agents.'
    )
    declare_social_bert_debug_scene_patch_every = DeclareLaunchArgument(
        'social_bert_debug_scene_patch_every',
        default_value=EnvironmentVariable('HUNAV_SOCIAL_BERT_DEBUG_SCENE_PATCH_EVERY', default_value='10'),
        description='Save one local map crop patch image every N SPU-BERT scene encodings.'
    )
    declare_social_bert_debug_model_io_image_dir = DeclareLaunchArgument(
        'social_bert_debug_model_io_image_dir',
        default_value=EnvironmentVariable('HUNAV_SOCIAL_BERT_DEBUG_MODEL_IO_IMAGE_DIR', default_value=''),
        description='If non-empty, save SPU-BERT input/output overview images to this directory.'
    )
    declare_social_bert_debug_model_io_image_agent_id = DeclareLaunchArgument(
        'social_bert_debug_model_io_image_agent_id',
        default_value=EnvironmentVariable('HUNAV_SOCIAL_BERT_DEBUG_MODEL_IO_IMAGE_AGENT_ID', default_value='1'),
        description='Agent id whose SPU-BERT model I/O overview image is saved; <=0 saves all agents.'
    )
    declare_social_bert_debug_model_io_image_every = DeclareLaunchArgument(
        'social_bert_debug_model_io_image_every',
        default_value=EnvironmentVariable('HUNAV_SOCIAL_BERT_DEBUG_MODEL_IO_IMAGE_EVERY', default_value='10'),
        description='Save one SPU-BERT model I/O overview image every N compute cycles.'
    )
    declare_jackal_spubert_controller = DeclareLaunchArgument(
        'jackal_spubert_controller',
        default_value=EnvironmentVariable('HUNAV_JACKAL_SPUBERT_CONTROLLER', default_value='False'),
        description='Launch experimental SPU-BERT local controller for Jackal on /cmd_vel.'
    )
    declare_robot_path_planner = DeclareLaunchArgument(
        'robot_path_planner',
        default_value=EnvironmentVariable('HUNAV_ROBOT_PATH_PLANNER', default_value='nav2'),
        description='PMB2 planner mode: nav2, monitor, or spubert.'
    )
    declare_robot_spubert_repo_path = DeclareLaunchArgument(
        'robot_spubert_repo_path',
        default_value=EnvironmentVariable(
            'HUNAV_ROBOT_SPUBERT_REPO_PATH',
            default_value='/home/hunav_gz_classic_ws/src/moai_social_bert_refactored'
        ),
        description='Guidance-conditioned moai_social_bert_refactored repository.'
    )
    declare_robot_spubert_config_path = DeclareLaunchArgument(
        'robot_spubert_config_path',
        default_value=EnvironmentVariable(
            'HUNAV_ROBOT_SPUBERT_CONFIG_PATH',
            default_value=(
                '/home/hunav_gz_classic_ws/src/moai_social_bert_refactored/'
                'configs/spubert/moai_social_nav_ext_scene_guided_fs.yaml'
            )
        ),
        description='Guidance-conditioned robot SPU-BERT YAML.'
    )
    declare_robot_spubert_checkpoint_path = DeclareLaunchArgument(
        'robot_spubert_checkpoint_path',
        default_value=EnvironmentVariable(
            'HUNAV_ROBOT_SPUBERT_CHECKPOINT',
            default_value=(
                '/home/hunav_gz_classic_ws/src/moai_social_bert_refactored/'
                'output/spubert_moai_gazebo_guided_mgp_fs/model_best.pth'
            )
        ),
        description='Fine-tuned guidance-conditioned robot SPU-BERT checkpoint.'
    )
    declare_robot_spubert_cuda = DeclareLaunchArgument(
        'robot_spubert_cuda',
        default_value=EnvironmentVariable('HUNAV_ROBOT_SPUBERT_CUDA', default_value='true'),
        description='Use CUDA for the PMB2 guided SPU-BERT planner.'
    )
    declare_robot_spubert_d_sample = DeclareLaunchArgument(
        'robot_spubert_d_sample',
        default_value=EnvironmentVariable('HUNAV_ROBOT_SPUBERT_D_SAMPLE', default_value='40'),
        description='MGP latent samples used for each online PMB2 replan.'
    )
    declare_robot_spubert_replan_period = DeclareLaunchArgument(
        'robot_spubert_replan_period',
        default_value=EnvironmentVariable('HUNAV_ROBOT_SPUBERT_REPLAN_PERIOD', default_value='0.8'),
        description='Seconds between SPU-BERT FollowPath updates.'
    )
    declare_robot_spubert_fallback_to_nav2 = DeclareLaunchArgument(
        'robot_spubert_fallback_to_nav2',
        default_value=EnvironmentVariable('HUNAV_ROBOT_SPUBERT_FALLBACK_NAV2', default_value='True'),
        description='Fall back to standard NavigateToPose when a learned path is unsafe.'
    )
    declare_robot_save_training_pkl = DeclareLaunchArgument(
        'robot_save_training_pkl',
        default_value=EnvironmentVariable('HUNAV_ROBOT_SAVE_TRAINING_PKL', default_value='False'),
        description='Record the robot target, surrounding humans, and active RViz goal as training windows.'
    )
    declare_robot_training_pkl_path = DeclareLaunchArgument(
        'robot_training_pkl_path',
        default_value=EnvironmentVariable(
            'HUNAV_ROBOT_TRAINING_PKL_PATH',
            default_value='/home/hunav_gz_classic_ws/moai_recordings/robot_target_all_trajs.pkl'
        ),
        description='Output path for robot-target Gazebo trajectory samples.'
    )
    declare_robot_training_record_dt = DeclareLaunchArgument(
        'robot_training_record_dt',
        default_value=EnvironmentVariable('HUNAV_ROBOT_TRAINING_RECORD_DT', default_value='0.4'),
        description='Seconds between recorded robot/human frames.'
    )
    declare_robot_training_sample_stride = DeclareLaunchArgument(
        'robot_training_sample_stride',
        default_value=EnvironmentVariable('HUNAV_ROBOT_TRAINING_SAMPLE_STRIDE', default_value='1'),
        description='Sliding-window stride in recorded frames.'
    )
    declare_robot_training_flush_every = DeclareLaunchArgument(
        'robot_training_flush_every',
        default_value=EnvironmentVariable('HUNAV_ROBOT_TRAINING_FLUSH_EVERY', default_value='10'),
        description='Persist the pkl after this many new samples.'
    )
    declare_robot_training_max_samples = DeclareLaunchArgument(
        'robot_training_max_samples',
        default_value=EnvironmentVariable('HUNAV_ROBOT_TRAINING_MAX_SAMPLES', default_value='0'),
        description='Maximum robot-target samples; 0 means unlimited.'
    )
    declare_auto_goal_enabled = DeclareLaunchArgument(
        'auto_goal_enabled',
        default_value=EnvironmentVariable('HUNAV_AUTO_GOAL', default_value='False'),
        description='Automatically publish validated /goal_pose episodes.'
    )
    declare_auto_goal_mode = DeclareLaunchArgument(
        'auto_goal_mode',
        default_value=EnvironmentVariable('HUNAV_AUTO_GOAL_MODE', default_value='waypoint'),
        description='Automatic goal policy: waypoint (basic curriculum) or random.'
    )
    declare_auto_goal_waypoints = DeclareLaunchArgument(
        'auto_goal_waypoints',
        default_value=EnvironmentVariable('HUNAV_AUTO_GOAL_WAYPOINTS', default_value=''),
        description="Optional waypoint override using 'x,y;x,y' format."
    )
    declare_auto_goal_seed = DeclareLaunchArgument(
        'auto_goal_seed',
        default_value=EnvironmentVariable('HUNAV_AUTO_GOAL_SEED', default_value='-1'),
        description='Random seed used by automatic goal generation; negative uses system randomness.'
    )
    declare_auto_goal_min_distance = DeclareLaunchArgument(
        'auto_goal_min_distance',
        default_value=EnvironmentVariable('HUNAV_AUTO_GOAL_MIN_DISTANCE', default_value='6.0'),
        description='Minimum robot-to-goal distance in meters.'
    )
    declare_auto_goal_max_distance = DeclareLaunchArgument(
        'auto_goal_max_distance',
        default_value=EnvironmentVariable('HUNAV_AUTO_GOAL_MAX_DISTANCE', default_value='20.0'),
        description='Maximum robot-to-goal distance in meters; 0 disables the maximum.'
    )
    declare_auto_goal_clearance = DeclareLaunchArgument(
        'auto_goal_clearance',
        default_value=EnvironmentVariable('HUNAV_AUTO_GOAL_CLEARANCE', default_value='0.55'),
        description='Required free-map clearance around each random goal in meters.'
    )
    declare_auto_goal_min_episode_duration = DeclareLaunchArgument(
        'auto_goal_min_episode_duration',
        default_value=EnvironmentVariable('HUNAV_AUTO_GOAL_MIN_EPISODE_DURATION', default_value='12.0'),
        description='Minimum seconds before replacing a reached goal.'
    )
    declare_auto_goal_timeout = DeclareLaunchArgument(
        'auto_goal_timeout',
        default_value=EnvironmentVariable('HUNAV_AUTO_GOAL_TIMEOUT', default_value='60.0'),
        description='Replace an unreached goal after this many seconds.'
    )
    declare_auto_goal_max_goals = DeclareLaunchArgument(
        'auto_goal_max_goals',
        default_value=EnvironmentVariable('HUNAV_AUTO_GOAL_MAX_GOALS', default_value='0'),
        description='Number of automatic goals; 0 means unlimited.'
    )
    declare_frame_to_publish = DeclareLaunchArgument(
        'global_frame_to_publish', default_value='map',
        description='Name of the global frame in which the position of the agents are provided'
    )
    declare_use_navgoal = DeclareLaunchArgument(
        'use_navgoal_to_start', default_value='False',
        description='Whether to start the agents movements when a navigation goal is received or not'
    )
    declare_navgoal_topic = DeclareLaunchArgument(
        'navgoal_topic', default_value='goal_pose',
        description='Name of the topic in which navigation goal for the robot will be published'
    )
    declare_navigation = DeclareLaunchArgument(
        'navigation', default_value='False',
        description='If launch the pmb2 navigation system'
    )
    declare_use_static_map_odom = DeclareLaunchArgument(
        'use_static_map_odom',
        default_value=EnvironmentVariable('HUNAV_USE_STATIC_MAP_ODOM', default_value='False'),
        description='Publish a fixed map->odom transform for simulation runs without AMCL laser localization'
    )
    declare_ignore_models = DeclareLaunchArgument(
        'ignore_models', default_value='aws_robomaker_warehouse_GroundB_01_001 ground_plane cafe Carpet_01_001 Carpet_01_002 Chandeliers_01_001 Chandeliers_01_002 Chandeliers_01_003 FloorB_01_001',
        description='list of Gazebo models that the agents should ignore as obstacles as the ground_plane. Indicate the models with a blank space between them'
    )
    declare_arg_verbose = DeclareLaunchArgument(
        'verbose', default_value='true',
        description='Set "true" to increase messages written to terminal.'
    )
    declare_arg_namespace = DeclareLaunchArgument('robot_namespace', default_value='',
            description='The type of robot')
    #DeclareLaunchArgument('gzpose', default_value='-x 0.0 -y 0.0 -z 0.1 -R 0.0 -P 0.0 -Y 1.57',
    #                      description='The robot initial position in the world')
    declare_arg_px = DeclareLaunchArgument(
            'gzpose_x',
            default_value=EnvironmentVariable('HUNAV_GZPOSE_X', default_value='0.0'),
            description='The robot initial position in the X axis of the world')
    declare_arg_py = DeclareLaunchArgument(
            'gzpose_y',
            default_value=EnvironmentVariable('HUNAV_GZPOSE_Y', default_value='0.0'),
            description='The robot initial position in the Y axis of the world')
    declare_arg_pz = DeclareLaunchArgument(
            'gzpose_z',
            default_value=EnvironmentVariable('HUNAV_GZPOSE_Z', default_value='0.25'),
            description='The robot initial position in the Z axis of the world')
    declare_arg_pR = DeclareLaunchArgument(
            'gzpose_R',
            default_value=EnvironmentVariable('HUNAV_GZPOSE_R', default_value='0.0'),
            description='The robot initial roll angle in the world')
    declare_arg_pP = DeclareLaunchArgument(
            'gzpose_P',
            default_value=EnvironmentVariable('HUNAV_GZPOSE_P', default_value='0.0'),
            description='The robot initial pitch angle in the world')
    declare_arg_pY = DeclareLaunchArgument(
            'gzpose_Y',
            default_value=EnvironmentVariable('HUNAV_GZPOSE_YAW', default_value='0.0'),
            description='The robot initial yaw angle in the world')
    declare_arg_laser = DeclareLaunchArgument('laser_model', default_value='sick-571-gpu',
            description='the laser model to be used')
    declare_arg_rgbd = DeclareLaunchArgument('rgbd_sensors', default_value='false',
            description='whether to use rgbd cameras or not')

    ld = LaunchDescription()

    # set environment variables
    ld.add_action(set_env_gazebo_model)
    ld.add_action(set_env_gazebo_resource)
    ld.add_action(set_env_gazebo_plugin)

    # Declare the launch arguments
    ld.add_action(declare_agents_conf_file)
    ld.add_action(declare_metrics_conf_file)
    ld.add_action(declare_arg_environment)
    ld.add_action(declare_gz_obs)
    ld.add_action(declare_update_rate)
    ld.add_action(declare_robot_type)
    ld.add_action(declare_use_gazebo_gui)
    ld.add_action(declare_use_rviz)
    ld.add_action(declare_use_hunav_evaluator)
    ld.add_action(declare_agent_motion_model)
    ld.add_action(declare_social_bert_predictor)
    ld.add_action(declare_spubert_model_path)
    ld.add_action(declare_spubert_repo_path)
    ld.add_action(declare_moai_spubert_repo_path)
    ld.add_action(declare_spubert_cuda)
    ld.add_action(declare_spubert_k_sample)
    ld.add_action(declare_spubert_d_sample)
    ld.add_action(declare_spubert_cache_ttl)
    ld.add_action(declare_spubert_refresh_dt)
    ld.add_action(declare_spubert_map_yaml_path)
    ld.add_action(declare_spubert_use_map_collision_filter)
    ld.add_action(declare_spubert_map_collision_radius)
    ld.add_action(declare_spubert_map_collision_weight)
    ld.add_action(declare_spubert_hard_map_collision_guard)
    ld.add_action(declare_spubert_candidate_selection_mode)
    ld.add_action(declare_spubert_guidance_point_radius)
    ld.add_action(declare_spubert_sync_on_cache_miss)
    ld.add_action(declare_social_bert_lookahead_step)
    ld.add_action(declare_social_bert_min_goal_speed)
    ld.add_action(declare_social_bert_max_speed)
    ld.add_action(declare_social_bert_max_accel)
    ld.add_action(declare_social_bert_goal_velocity_blend)
    ld.add_action(declare_social_bert_goal_arrival_distance)
    ld.add_action(declare_social_bert_goal_arrival_min_speed)
    ld.add_action(declare_social_bert_robot_personal_space)
    ld.add_action(declare_social_bert_robot_collision_buffer)
    ld.add_action(declare_social_bert_robot_hard_collision_guard)
    ld.add_action(declare_social_bert_agent_personal_space)
    ld.add_action(declare_social_bert_agent_avoidance_gain)
    ld.add_action(declare_social_bert_obstacle_avoidance_distance)
    ld.add_action(declare_social_bert_obstacle_avoidance_gain)
    ld.add_action(declare_social_bert_obstacle_collision_buffer)
    ld.add_action(declare_social_bert_obstacle_lateral_speed_ratio)
    ld.add_action(declare_social_bert_velocity_smoothing_alpha)
    ld.add_action(declare_social_bert_collision_buffer)
    ld.add_action(declare_social_bert_max_lateral_speed_ratio)
    ld.add_action(declare_social_bert_max_yaw_rate)
    ld.add_action(declare_social_bert_controller_avoidance_enabled)
    ld.add_action(declare_social_bert_save_training_pkl)
    ld.add_action(declare_social_bert_training_pkl_path)
    ld.add_action(declare_social_bert_training_record_dt)
    ld.add_action(declare_social_bert_training_sample_stride)
    ld.add_action(declare_social_bert_training_flush_every)
    ld.add_action(declare_social_bert_training_max_samples)
    ld.add_action(declare_social_bert_publish_debug_markers)
    ld.add_action(declare_social_bert_debug_marker_publish_every)
    ld.add_action(declare_social_bert_debug_focus_agent_id)
    ld.add_action(declare_social_bert_debug_focus_agent_only)
    ld.add_action(declare_social_bert_debug_guidance_only)
    ld.add_action(declare_social_bert_debug_show_all_candidate_paths)
    ld.add_action(declare_social_bert_debug_show_all_model_io)
    ld.add_action(declare_social_bert_debug_scene_patch_dir)
    ld.add_action(declare_social_bert_debug_scene_patch_agent_id)
    ld.add_action(declare_social_bert_debug_scene_patch_every)
    ld.add_action(declare_social_bert_debug_model_io_image_dir)
    ld.add_action(declare_social_bert_debug_model_io_image_agent_id)
    ld.add_action(declare_social_bert_debug_model_io_image_every)
    ld.add_action(declare_jackal_spubert_controller)
    ld.add_action(declare_robot_path_planner)
    ld.add_action(declare_robot_spubert_repo_path)
    ld.add_action(declare_robot_spubert_config_path)
    ld.add_action(declare_robot_spubert_checkpoint_path)
    ld.add_action(declare_robot_spubert_cuda)
    ld.add_action(declare_robot_spubert_d_sample)
    ld.add_action(declare_robot_spubert_replan_period)
    ld.add_action(declare_robot_spubert_fallback_to_nav2)
    ld.add_action(declare_robot_save_training_pkl)
    ld.add_action(declare_robot_training_pkl_path)
    ld.add_action(declare_robot_training_record_dt)
    ld.add_action(declare_robot_training_sample_stride)
    ld.add_action(declare_robot_training_flush_every)
    ld.add_action(declare_robot_training_max_samples)
    ld.add_action(declare_auto_goal_enabled)
    ld.add_action(declare_auto_goal_mode)
    ld.add_action(declare_auto_goal_waypoints)
    ld.add_action(declare_auto_goal_seed)
    ld.add_action(declare_auto_goal_min_distance)
    ld.add_action(declare_auto_goal_max_distance)
    ld.add_action(declare_auto_goal_clearance)
    ld.add_action(declare_auto_goal_min_episode_duration)
    ld.add_action(declare_auto_goal_timeout)
    ld.add_action(declare_auto_goal_max_goals)
    ld.add_action(declare_robot_name)
    ld.add_action(declare_frame_to_publish)
    ld.add_action(declare_use_navgoal)
    ld.add_action(declare_navgoal_topic)
    ld.add_action(declare_navigation)
    ld.add_action(declare_use_static_map_odom)
    ld.add_action(declare_ignore_models)
    ld.add_action(declare_arg_verbose)
    ld.add_action(declare_arg_namespace)
    ld.add_action(declare_arg_laser)
    ld.add_action(declare_arg_rgbd)
    ld.add_action(declare_arg_px)
    ld.add_action(declare_arg_py)
    ld.add_action(declare_arg_pz)
    ld.add_action(declare_arg_pR)
    ld.add_action(declare_arg_pP)
    ld.add_action(declare_arg_pY)

    # Generate the world with the agents
    # launch hunav_loader and the WorldGenerator
    # 2 seconds later
    ld.add_action(hunav_loader_node)
    ld.add_action(ordered_launch_event)

    # hunav behavior manager node
    ld.add_action(manager_launch_event)
    # hunav evaluator
    ld.add_action(hunav_evaluator_node)
    ld.add_action(human_obstacle_cloud_node)
    ld.add_action(pmb2_cmd_vel_passthrough_node)
    ld.add_action(spubert_nav2_bridge_node)
    ld.add_action(spubert_jackal_controller_node)
    ld.add_action(robot_dataset_logger_node)
    ld.add_action(random_goal_publisher_node)

    # launch Gazebo after worldGenerator 
    ld.add_action(gz_launch_event)
    ld.add_action(static_tf_node)
    ld.add_action(ground_truth_localization_node)

    # spawn robot in Gazebo after gzserver has started
    ld.add_action(robot_launch_event)

    return ld

    


# Add boolean commands if true
def _boolean_command(arg):
    cmd = ['"--', arg, '" if "true" == "', LaunchConfiguration(arg), '" else ""']
    py_cmd = PythonExpression(cmd)
    return py_cmd
