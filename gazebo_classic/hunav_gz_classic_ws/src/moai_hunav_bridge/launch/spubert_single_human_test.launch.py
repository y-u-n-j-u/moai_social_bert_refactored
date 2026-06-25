from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    agent_motion_model = LaunchConfiguration("agent_motion_model")
    predictor = LaunchConfiguration("predictor")
    spubert_model_path = LaunchConfiguration("spubert_model_path")
    spubert_repo_path = LaunchConfiguration("spubert_repo_path")
    moai_spubert_repo_path = LaunchConfiguration("moai_spubert_repo_path")
    spubert_candidate_selection_mode = LaunchConfiguration("spubert_candidate_selection_mode")
    controller_avoidance_enabled = LaunchConfiguration("controller_avoidance_enabled")
    spubert_debug_model_io_image_dir = LaunchConfiguration("spubert_debug_model_io_image_dir")
    spubert_debug_model_io_image_agent_id = LaunchConfiguration("spubert_debug_model_io_image_agent_id")
    spubert_debug_model_io_image_every = LaunchConfiguration("spubert_debug_model_io_image_every")
    use_rviz = LaunchConfiguration("use_rviz")
    save_training_pkl = LaunchConfiguration("save_training_pkl")
    training_pkl_path = LaunchConfiguration("training_pkl_path")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_rviz",
                default_value="true",
                description="Open RViz with human history, predicted path, and goal markers.",
            ),
            DeclareLaunchArgument(
                "predictor",
                default_value=EnvironmentVariable("HUNAV_SOCIAL_BERT_PREDICTOR", default_value="constant_velocity"),
                description="Human motion predictor: constant_velocity, spubert, or moai_spubert.",
            ),
            DeclareLaunchArgument(
                "agent_motion_model",
                default_value="spubert",
                description="Pedestrian motion model: hunav for default HuNavSim social force, spubert for the MOAI SPU-BERT predictor bridge.",
            ),
            DeclareLaunchArgument(
                "spubert_model_path",
                default_value=EnvironmentVariable("HUNAV_SPUBERT_MODEL_PATH", default_value=""),
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
                "spubert_candidate_selection_mode",
                default_value="map_goal",
                description="Candidate selector: goal, map_goal, or safety_goal.",
            ),
            DeclareLaunchArgument(
                "controller_avoidance_enabled",
                default_value="false",
                description="Apply controller-level social/obstacle avoidance after the selected SPU-BERT path.",
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
                "save_training_pkl",
                default_value="false",
                description="Save Gazebo trajectories as moai all_trajs pkl samples.",
            ),
            DeclareLaunchArgument(
                "training_pkl_path",
                default_value="/tmp/moai_gazebo_single_human_all_trajs.pkl",
                description="Output path for recorded moai all_trajs pkl.",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [FindPackageShare("hunav_gazebo_wrapper"), "launch", "simulation.launch.py"]
                    )
                ),
                launch_arguments={
                    "environment_name": "simple_corridor",
                    "configuration_file": "agents_single_corridor.yaml",
                    "robot_type": "jackal",
                    "robot_name": "jackal",
                    "navigation": "False",
                    "agent_motion_model": agent_motion_model,
                    "social_bert_predictor": predictor,
                    "spubert_model_path": spubert_model_path,
                    "spubert_repo_path": spubert_repo_path,
                    "moai_spubert_repo_path": moai_spubert_repo_path,
                    "spubert_candidate_selection_mode": spubert_candidate_selection_mode,
                    "jackal_spubert_controller": "False",
                    "use_gazebo_obs": "True",
                    "update_rate": "20.0",
                    "gzpose_x": "-21.0",
                    "gzpose_y": "-3.5",
                    "gzpose_Y": "0.0",
                    "social_bert_max_speed": "1.35",
                    "social_bert_max_accel": "1.6",
                    "social_bert_lookahead_step": "3",
                    "social_bert_min_goal_speed": "0.65",
                    "social_bert_goal_velocity_blend": "0.55",
                    "social_bert_agent_personal_space": "0.9",
                    "social_bert_obstacle_avoidance_distance": "1.8",
                    "social_bert_obstacle_avoidance_gain": "0.75",
                    "social_bert_obstacle_collision_buffer": "0.65",
                    "social_bert_obstacle_lateral_speed_ratio": "0.85",
                    "social_bert_max_lateral_speed_ratio": "0.2",
                    "social_bert_max_yaw_rate": "0.9",
                    "social_bert_controller_avoidance_enabled": controller_avoidance_enabled,
                    "social_bert_debug_model_io_image_dir": spubert_debug_model_io_image_dir,
                    "social_bert_debug_model_io_image_agent_id": spubert_debug_model_io_image_agent_id,
                    "social_bert_debug_model_io_image_every": spubert_debug_model_io_image_every,
                    "social_bert_save_training_pkl": save_training_pkl,
                    "social_bert_training_pkl_path": training_pkl_path,
                    "social_bert_training_record_dt": "0.4",
                    "social_bert_training_sample_stride": "1",
                    "social_bert_training_flush_every": "10",
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
                package="rviz2",
                executable="rviz2",
                name="spubert_single_human_debug_rviz",
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
