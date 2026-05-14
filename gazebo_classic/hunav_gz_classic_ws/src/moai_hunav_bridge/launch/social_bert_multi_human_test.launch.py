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
    spubert_d_sample = LaunchConfiguration("spubert_d_sample")
    predictor = LaunchConfiguration("predictor")
    spubert_model_path = LaunchConfiguration("spubert_model_path")
    moai_spubert_repo_path = LaunchConfiguration("moai_spubert_repo_path")
    spubert_map_yaml_path = LaunchConfiguration("spubert_map_yaml_path")
    spubert_use_map_collision_filter = LaunchConfiguration("spubert_use_map_collision_filter")
    spubert_map_collision_radius = LaunchConfiguration("spubert_map_collision_radius")
    spubert_map_collision_weight = LaunchConfiguration("spubert_map_collision_weight")
    spubert_debug_scene_patch_dir = LaunchConfiguration("spubert_debug_scene_patch_dir")
    spubert_debug_scene_patch_agent_id = LaunchConfiguration("spubert_debug_scene_patch_agent_id")
    spubert_debug_scene_patch_every = LaunchConfiguration("spubert_debug_scene_patch_every")
    debug_focus_agent_id = LaunchConfiguration("debug_focus_agent_id")
    debug_show_all_candidate_paths = LaunchConfiguration("debug_show_all_candidate_paths")
    debug_show_all_model_io = LaunchConfiguration("debug_show_all_model_io")
    use_rviz = LaunchConfiguration("use_rviz")
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
                "environment_name",
                default_value="simple_corridor",
                description="HuNavSim/Gazebo environment name, for example simple_corridor, cafe, house, or warehouse.",
            ),
            DeclareLaunchArgument(
                "configuration_file",
                default_value="agents_simple_corridor_5.yaml",
                description="HuNavSim scenario YAML file, for example agents_simple_corridor_5.yaml or agents_cafe.yaml.",
            ),
            DeclareLaunchArgument(
                "update_rate",
                default_value="10.0",
                description="HuNavSim update rate. Lower this for heavier scenarios such as cafe with 10 agents.",
            ),
            DeclareLaunchArgument(
                "spubert_d_sample",
                default_value="20",
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
                "debug_focus_agent_id",
                default_value="1",
                description="Agent id whose SPU-BERT input/output markers are expanded in RViz; <=0 shows all.",
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
                "moai_spubert_repo_path",
                default_value=EnvironmentVariable(
                    "HUNAV_MOAI_SPUBERT_REPO_PATH",
                    default_value="/home/hunav_gz_classic_ws/src/moai_social_bert_refactored",
                ),
                description="Path to moai_social_bert_refactored inside the container.",
            ),
            DeclareLaunchArgument(
                "save_training_pkl",
                default_value="true",
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
                    "robot_type": "jackal",
                    "robot_name": "jackal",
                    "navigation": "False",
                    "agent_motion_model": "social_bert",
                    "social_bert_predictor": predictor,
                    "spubert_model_path": spubert_model_path,
                    "moai_spubert_repo_path": moai_spubert_repo_path,
                    "spubert_k_sample": "20",
                    "spubert_d_sample": spubert_d_sample,
                    "spubert_cache_ttl": "1.5",
                    "spubert_map_yaml_path": spubert_map_yaml_path,
                    "spubert_use_map_collision_filter": spubert_use_map_collision_filter,
                    "spubert_map_collision_radius": spubert_map_collision_radius,
                    "spubert_map_collision_weight": spubert_map_collision_weight,
                    "social_bert_debug_scene_patch_dir": spubert_debug_scene_patch_dir,
                    "social_bert_debug_scene_patch_agent_id": spubert_debug_scene_patch_agent_id,
                    "social_bert_debug_scene_patch_every": spubert_debug_scene_patch_every,
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
                    "social_bert_agent_personal_space": "1.05",
                    "social_bert_agent_avoidance_gain": "1.0",
                    "social_bert_obstacle_avoidance_distance": "1.8",
                    "social_bert_obstacle_avoidance_gain": "0.7",
                    "social_bert_obstacle_collision_buffer": "0.55",
                    "social_bert_obstacle_lateral_speed_ratio": "1.4",
                    "social_bert_max_lateral_speed_ratio": "0.8",
                    "social_bert_max_yaw_rate": "1.1",
                    "social_bert_save_training_pkl": save_training_pkl,
                    "social_bert_training_pkl_path": training_pkl_path,
                    "social_bert_training_record_dt": "0.4",
                    "social_bert_training_sample_stride": "1",
                    "social_bert_training_flush_every": "10",
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
                parameters=[
                    {
                        "use_sim_time": True,
                        "human_states_topic": "/human_states",
                        "marker_topic": "/moai/debug_human_paths",
                        "obs_len": 8,
                        "pred_len": 12,
                        "prediction_dt": 0.4,
                        "history_timeout": 3.0,
                        "publish_rate": 5.0,
                    }
                ],
            ),
            Node(
                package="moai_hunav_bridge",
                executable="human_obstacle_cloud_node",
                name="human_obstacle_cloud",
                output="screen",
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
                parameters=[
                    {
                        "use_sim_time": True,
                        "enabled": save_jackal_training_pkl,
                        "robot_topic": "/robot_states",
                        "human_states_topic": "/human_states",
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
                name="social_bert_multi_human_debug_rviz",
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
