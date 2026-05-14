from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    predictor = LaunchConfiguration("predictor")
    spubert_model_path = LaunchConfiguration("spubert_model_path")
    moai_spubert_repo_path = LaunchConfiguration("moai_spubert_repo_path")
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
                "spubert_model_path",
                default_value=EnvironmentVariable("HUNAV_SPUBERT_MODEL_PATH", default_value=""),
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
                    "agent_motion_model": "social_bert",
                    "social_bert_predictor": predictor,
                    "spubert_model_path": spubert_model_path,
                    "moai_spubert_repo_path": moai_spubert_repo_path,
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
                name="social_bert_human_debug_rviz",
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
