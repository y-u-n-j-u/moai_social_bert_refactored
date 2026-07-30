from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    environment_name = LaunchConfiguration("environment_name")
    configuration_file = LaunchConfiguration("configuration_file")
    update_rate = LaunchConfiguration("update_rate")
    agent_motion_model = LaunchConfiguration("agent_motion_model")
    robot_type = LaunchConfiguration("robot_type")
    robot_name = LaunchConfiguration("robot_name")
    use_gazebo_gui = LaunchConfiguration("use_gazebo_gui")
    use_hunav_evaluator = LaunchConfiguration("use_hunav_evaluator")
    spubert_k_sample = LaunchConfiguration("spubert_k_sample")
    spubert_d_sample = LaunchConfiguration("spubert_d_sample")
    predictor = LaunchConfiguration("predictor")
    spubert_model_path = LaunchConfiguration("spubert_model_path")
    spubert_repo_path = LaunchConfiguration("spubert_repo_path")
    moai_spubert_repo_path = LaunchConfiguration("moai_spubert_repo_path")
    spubert_map_yaml_path = LaunchConfiguration("spubert_map_yaml_path")
    spubert_use_map_collision_filter = LaunchConfiguration("spubert_use_map_collision_filter")
    spubert_map_collision_radius = LaunchConfiguration("spubert_map_collision_radius")
    spubert_map_collision_weight = LaunchConfiguration("spubert_map_collision_weight")
    spubert_hard_map_collision_guard = LaunchConfiguration("spubert_hard_map_collision_guard")
    spubert_candidate_selection_mode = LaunchConfiguration("spubert_candidate_selection_mode")
    spubert_guidance_point_radius = LaunchConfiguration("spubert_guidance_point_radius")
    spubert_sync_on_cache_miss = LaunchConfiguration("spubert_sync_on_cache_miss")
    controller_avoidance_enabled = LaunchConfiguration("controller_avoidance_enabled")
    spubert_cache_ttl = LaunchConfiguration("spubert_cache_ttl")
    spubert_refresh_dt = LaunchConfiguration("spubert_refresh_dt")
    spubert_debug_scene_patch_dir = LaunchConfiguration("spubert_debug_scene_patch_dir")
    spubert_debug_scene_patch_agent_id = LaunchConfiguration("spubert_debug_scene_patch_agent_id")
    spubert_debug_scene_patch_every = LaunchConfiguration("spubert_debug_scene_patch_every")
    spubert_debug_model_io_image_dir = LaunchConfiguration("spubert_debug_model_io_image_dir")
    spubert_debug_model_io_image_agent_id = LaunchConfiguration("spubert_debug_model_io_image_agent_id")
    spubert_debug_model_io_image_every = LaunchConfiguration("spubert_debug_model_io_image_every")
    debug_focus_agent_id = LaunchConfiguration("debug_focus_agent_id")
    publish_debug_markers = LaunchConfiguration("publish_debug_markers")
    debug_marker_publish_every = LaunchConfiguration("debug_marker_publish_every")
    debug_show_all_candidate_paths = LaunchConfiguration("debug_show_all_candidate_paths")
    debug_show_all_model_io = LaunchConfiguration("debug_show_all_model_io")
    use_rviz = LaunchConfiguration("use_rviz")
    use_human_path_debug = LaunchConfiguration("use_human_path_debug")
    human_path_debug_publish_rate = LaunchConfiguration("human_path_debug_publish_rate")
    use_human_obstacle_cloud = LaunchConfiguration("use_human_obstacle_cloud")
    save_training_pkl = LaunchConfiguration("save_training_pkl")
    training_pkl_path = LaunchConfiguration("training_pkl_path")
    save_jackal_training_pkl = LaunchConfiguration("save_jackal_training_pkl")
    jackal_training_pkl_path = LaunchConfiguration("jackal_training_pkl_path")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_rviz",
                default_value="true",
                description="Open RViz with human history, predicted path, and obstacle markers.",
            ),
            DeclareLaunchArgument(
                "use_gazebo_gui",
                default_value="true",
                description="Open gzclient. Set false for headless/faster SPU-BERT pedestrian tests.",
            ),
            DeclareLaunchArgument(
                "use_hunav_evaluator",
                default_value="false",
                description="Run HuNav evaluator metrics node. Disable for lighter SPU-BERT motion debugging.",
            ),
            DeclareLaunchArgument(
                "use_human_path_debug",
                default_value="false",
                description="Run the extra human path debug marker node. Disable for smoother Gazebo/RViz playback.",
            ),
            DeclareLaunchArgument(
                "human_path_debug_publish_rate",
                default_value="2.0",
                description="Publish rate for the optional observed-history RViz debug node.",
            ),
            DeclareLaunchArgument(
                "use_human_obstacle_cloud",
                default_value="false",
                description="Publish predicted humans as a PointCloud obstacle layer. Not needed when Jackal lidar/Nav2 is disabled.",
            ),
            DeclareLaunchArgument(
                "environment_name",
                default_value="training_corridor",
                description="HuNavSim/Gazebo training environment name.",
            ),
            DeclareLaunchArgument(
                "configuration_file",
                default_value="agents_training_corridor_medium.yaml",
                description="HuNavSim low, medium, or high density training scenario YAML.",
            ),
            DeclareLaunchArgument(
                "update_rate",
                default_value="10.0",
                description="HuNavSim update rate. Lower this for high-density scenarios.",
            ),
            DeclareLaunchArgument(
                "agent_motion_model",
                default_value="spubert",
                description="Pedestrian motion model: hunav for default HuNavSim social force, spubert for the MOAI SPU-BERT predictor bridge.",
            ),
            DeclareLaunchArgument(
                "robot_type",
                default_value="none",
                description="Robot launch profile: none, jackal, or pmb2. Use none for pedestrian-only SPU-BERT tests.",
            ),
            DeclareLaunchArgument(
                "robot_name",
                default_value="none",
                description="Robot model name. Use jackal with robot_type:=jackal.",
            ),
            DeclareLaunchArgument(
                "spubert_k_sample",
                default_value="10",
                description="Number of SPU-BERT final trajectory candidates.",
            ),
            DeclareLaunchArgument(
                "spubert_d_sample",
                default_value="10",
                description="Number of SPU-BERT goal samples. Lower values are faster for crowded scenarios.",
            ),
            DeclareLaunchArgument(
                "spubert_map_yaml_path",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("hunav_gazebo_wrapper"),
                        "maps",
                        PythonExpression(["'", environment_name, ".yaml'"]),
                    ]
                ),
                description="ROS map YAML used as local occupancy crop input for SPU-BERT pedestrians.",
            ),
            DeclareLaunchArgument(
                "spubert_use_map_collision_filter",
                default_value="true",
                description="Penalize SPU-BERT candidate paths that collide with the occupancy map.",
            ),
            DeclareLaunchArgument(
                "spubert_map_collision_radius",
                default_value="0.35",
                description="Pedestrian radius used for map collision checking of SPU-BERT candidates.",
            ),
            DeclareLaunchArgument(
                "spubert_map_collision_weight",
                default_value="100.0",
                description="Penalty per colliding waypoint when selecting a SPU-BERT candidate path.",
            ),
            DeclareLaunchArgument(
                "spubert_hard_map_collision_guard",
                default_value="false",
                description="Block the next pedestrian step if it would enter an occupied map cell.",
            ),
            DeclareLaunchArgument(
                "spubert_candidate_selection_mode",
                default_value="guidance_point",
                description="Candidate selector: guidance_point, final_goal, or none.",
            ),
            DeclareLaunchArgument(
                "spubert_guidance_point_radius",
                default_value="8.0",
                description="Radius in meters for current-centered pedestrian guidance point.",
            ),
            DeclareLaunchArgument(
                "spubert_sync_on_cache_miss",
                default_value="true",
                description="Run one blocking SPU-BERT inference when no cached candidate paths are available.",
            ),
            DeclareLaunchArgument(
                "controller_avoidance_enabled",
                default_value="false",
                description="Apply controller-level social/obstacle avoidance after the selected SPU-BERT path.",
            ),
            DeclareLaunchArgument(
                "spubert_cache_ttl",
                default_value="2.5",
                description="Seconds to reuse cached SPU-BERT predictions. Keep this >= spubert_refresh_dt for smoother playback.",
            ),
            DeclareLaunchArgument(
                "spubert_refresh_dt",
                default_value="1.0",
                description="Seconds between fresh SPU-BERT inferences per agent. Increase this to reduce lag.",
            ),
            DeclareLaunchArgument(
                "spubert_debug_scene_patch_dir",
                default_value="",
                description="If non-empty, save target-centered SPU-BERT occupancy crop patch-grid images here.",
            ),
            DeclareLaunchArgument(
                "spubert_debug_scene_patch_agent_id",
                default_value="1",
                description="Agent id whose SPU-BERT map crop patch image is saved; <=0 saves all agents.",
            ),
            DeclareLaunchArgument(
                "spubert_debug_scene_patch_every",
                default_value="10",
                description="Save one SPU-BERT map crop patch image every N scene encodings.",
            ),
            DeclareLaunchArgument(
                "spubert_debug_model_io_image_dir",
                default_value="",
                description="If non-empty, save SPU-BERT input/output overview images here.",
            ),
            DeclareLaunchArgument(
                "spubert_debug_model_io_image_agent_id",
                default_value="1",
                description="Agent id whose SPU-BERT input/output overview image is saved; <=0 saves all agents.",
            ),
            DeclareLaunchArgument(
                "spubert_debug_model_io_image_every",
                default_value="10",
                description="Save one SPU-BERT input/output overview image every N compute cycles.",
            ),
            DeclareLaunchArgument(
                "debug_focus_agent_id",
                default_value="1",
                description="Agent id whose SPU-BERT input/output markers are expanded in RViz; <=0 shows all.",
            ),
            DeclareLaunchArgument(
                "publish_debug_markers",
                default_value="true",
                description="Publish SPU-BERT RViz marker arrays. Disable for smoother Gazebo playback.",
            ),
            DeclareLaunchArgument(
                "debug_marker_publish_every",
                default_value="3",
                description="Publish RViz debug markers every N compute cycles to reduce visual stutter.",
            ),
            DeclareLaunchArgument(
                "debug_show_all_candidate_paths",
                default_value="False",
                description="Show all SPU-BERT candidate path bundles in RViz instead of only the focus agent.",
            ),
            DeclareLaunchArgument(
                "debug_show_all_model_io",
                default_value="False",
                description="Show numbered obs/pred model input-output markers for every agent.",
            ),
            DeclareLaunchArgument(
                "predictor",
                default_value=EnvironmentVariable("HUNAV_SOCIAL_BERT_PREDICTOR", default_value="spubert"),
                description="Human motion predictor: constant_velocity, spubert, or moai_spubert.",
            ),
            DeclareLaunchArgument(
                "spubert_model_path",
                default_value=EnvironmentVariable(
                    "HUNAV_SPUBERT_MODEL_PATH",
                    default_value="/home/hunav_gz_classic_ws/src/moai_social_bert_refactored/output/ethucy/univ/spubert.pth",
                ),
                description="Checkpoint path used when predictor is spubert or moai_spubert.",
            ),
            DeclareLaunchArgument(
                "spubert_repo_path",
                default_value="/home/hunav_gz_classic_ws/src/moai_social_bert_refactored/runtime/SPUBERT",
                description="Path to the SPU-BERT runtime code used to load the checkpoint.",
            ),
            DeclareLaunchArgument(
                "moai_spubert_repo_path",
                default_value=EnvironmentVariable(
                    "HUNAV_MOAI_SPUBERT_REPO_PATH",
                    default_value="/home/hunav_gz_classic_ws/src/moai_social_bert_refactored",
                ),
                description="Path to moai_social_bert_refactored inside the container.",
            ),
            DeclareLaunchArgument(
                "save_training_pkl",
                default_value="false",
                description="Save multi-human Gazebo trajectories as moai all_trajs pkl samples.",
            ),
            DeclareLaunchArgument(
                "training_pkl_path",
                default_value="/tmp/moai_gazebo_multi_human_all_trajs.pkl",
                description="Output path for recorded moai all_trajs pkl.",
            ),
            DeclareLaunchArgument(
                "save_jackal_training_pkl",
                default_value="false",
                description="Save Jackal-as-target teleop trajectories with humans as neighbors.",
            ),
            DeclareLaunchArgument(
                "jackal_training_pkl_path",
                default_value="/home/hunav_gz_classic_ws/moai_recordings/jackal_teleop_all_trajs.pkl",
                description="Output path for Jackal-target MOAI all_trajs pkl.",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [FindPackageShare("hunav_gazebo_wrapper"), "launch", "simulation.launch.py"]
                    )
                ),
                launch_arguments={
                    "environment_name": environment_name,
                    "configuration_file": configuration_file,
                    "robot_type": robot_type,
                    "robot_name": robot_name,
                    "use_gazebo_gui": use_gazebo_gui,
                    "use_hunav_evaluator": use_hunav_evaluator,
                    "navigation": "False",
                    "agent_motion_model": agent_motion_model,
                    "social_bert_predictor": predictor,
                    "spubert_model_path": spubert_model_path,
                    "spubert_repo_path": spubert_repo_path,
                    "moai_spubert_repo_path": moai_spubert_repo_path,
                    "spubert_k_sample": spubert_k_sample,
                    "spubert_d_sample": spubert_d_sample,
                    "spubert_cache_ttl": spubert_cache_ttl,
                    "spubert_refresh_dt": spubert_refresh_dt,
                    "spubert_map_yaml_path": spubert_map_yaml_path,
                    "spubert_use_map_collision_filter": spubert_use_map_collision_filter,
                    "spubert_map_collision_radius": spubert_map_collision_radius,
                    "spubert_map_collision_weight": spubert_map_collision_weight,
                    "spubert_hard_map_collision_guard": spubert_hard_map_collision_guard,
                    "spubert_candidate_selection_mode": spubert_candidate_selection_mode,
                    "spubert_guidance_point_radius": spubert_guidance_point_radius,
                    "spubert_sync_on_cache_miss": spubert_sync_on_cache_miss,
                    "social_bert_debug_scene_patch_dir": spubert_debug_scene_patch_dir,
                    "social_bert_debug_scene_patch_agent_id": spubert_debug_scene_patch_agent_id,
                    "social_bert_debug_scene_patch_every": spubert_debug_scene_patch_every,
                    "social_bert_debug_model_io_image_dir": spubert_debug_model_io_image_dir,
                    "social_bert_debug_model_io_image_agent_id": spubert_debug_model_io_image_agent_id,
                    "social_bert_debug_model_io_image_every": spubert_debug_model_io_image_every,
                    "spubert_cuda": "true",
                    "jackal_spubert_controller": "False",
                    "use_gazebo_obs": "True",
                    "update_rate": update_rate,
                    "gzpose_x": "-21.0",
                    "gzpose_y": "-4.8",
                    "gzpose_Y": "0.0",
                    "social_bert_max_speed": "1.35",
                    "social_bert_max_accel": "2.2",
                    "social_bert_lookahead_step": "3",
                    "social_bert_min_goal_speed": "0.55",
                    "social_bert_goal_velocity_blend": "0.65",
                    "social_bert_goal_arrival_distance": "1.5",
                    "social_bert_goal_arrival_min_speed": "0.08",
                    "social_bert_agent_personal_space": "1.05",
                    "social_bert_agent_avoidance_gain": "1.0",
                    "social_bert_obstacle_avoidance_distance": "1.8",
                    "social_bert_obstacle_avoidance_gain": "0.7",
                    "social_bert_obstacle_collision_buffer": "0.55",
                    "social_bert_obstacle_lateral_speed_ratio": "1.4",
                    "social_bert_max_lateral_speed_ratio": "0.8",
                    "social_bert_max_yaw_rate": "1.1",
                    "social_bert_controller_avoidance_enabled": controller_avoidance_enabled,
                    "social_bert_save_training_pkl": save_training_pkl,
                    "social_bert_training_pkl_path": training_pkl_path,
                    "social_bert_training_record_dt": "0.4",
                    "social_bert_training_sample_stride": "1",
                    "social_bert_training_flush_every": "10",
                    "social_bert_publish_debug_markers": publish_debug_markers,
                    "social_bert_debug_marker_publish_every": debug_marker_publish_every,
                    "social_bert_debug_focus_agent_id": debug_focus_agent_id,
                    "social_bert_debug_show_all_candidate_paths": debug_show_all_candidate_paths,
                    "social_bert_debug_show_all_model_io": debug_show_all_model_io,
                }.items(),
            ),
            Node(
                package="moai_hunav_bridge",
                executable="human_path_debug_node",
                name="human_path_debug_node",
                output="screen",
                condition=IfCondition(use_human_path_debug),
                parameters=[
                    {
                        "use_sim_time": True,
                        "human_states_topic": "/human_states",
                        "marker_topic": "/moai/debug_human_paths",
                        "obs_len": 8,
                        "pred_len": 12,
                        "prediction_dt": 0.4,
                        "history_timeout": 3.0,
                        "publish_rate": human_path_debug_publish_rate,
                    }
                ],
            ),
            Node(
                package="moai_hunav_bridge",
                executable="human_obstacle_cloud_node",
                name="human_obstacle_cloud",
                output="screen",
                condition=IfCondition(use_human_obstacle_cloud),
                parameters=[
                    {
                        "use_sim_time": True,
                        "human_states_topic": "/human_states",
                        "predicted_paths_topic": "/moai/social_bert_predicted_paths",
                        "cloud_topic": "/moai/human_obstacle_cloud",
                    }
                ],
            ),
            Node(
                package="moai_hunav_bridge",
                executable="jackal_teleop_dataset_logger_node",
                name="jackal_teleop_dataset_logger",
                output="screen",
                condition=IfCondition(save_jackal_training_pkl),
                parameters=[
                    {
                        "use_sim_time": True,
                        "enabled": save_jackal_training_pkl,
                        "robot_topic": "/robot_states",
                        "human_states_topic": "/human_states",
                        "goal_topic": "/goal_pose",
                        "require_goal": True,
                        "guidance_point_radius": spubert_guidance_point_radius,
                        "output_path": jackal_training_pkl_path,
                        "obs_len": 8,
                        "pred_len": 12,
                        "record_dt": 0.4,
                        "sample_stride": 1,
                        "flush_every": 10,
                        "stale_timeout": 2.0,
                    }
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="spubert_multi_human_debug_rviz",
                output="screen",
                arguments=[
                    "-d",
                    PathJoinSubstitution(
                        [FindPackageShare("moai_hunav_bridge"), "rviz", "social_bert_human_debug.rviz"]
                    ),
                ],
                condition=IfCondition(use_rviz),
            ),
        ]
    )
